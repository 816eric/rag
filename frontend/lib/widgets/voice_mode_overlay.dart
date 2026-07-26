import 'dart:async';
import 'dart:typed_data';

import 'package:audioplayers/audioplayers.dart';
import 'package:flutter/material.dart';
import 'package:record/record.dart';

import '../api_client.dart';
import '../audio/wav_encoder.dart';
import '../theme.dart';

enum _VoiceState { listening, userSpeaking, transcribing, thinking, aiSpeaking }

const int _kSampleRate = 16000;
// Amplitude is dBFS from record's own meter (~-160 silence to 0 max). A real
// microphone's natural level varies hugely by hardware and room - a fixed
// threshold tuned against one signal (e.g. a loud synthetic test WAV) can
// simply never be crossed by actual quiet speech. Instead, track a rolling
// noise floor from recent ambient samples and require speech to be clearly
// above THAT, so this adapts to whatever mic/room it's actually running in.
const double _kSpeechMarginDb = 10; // how much louder than ambient counts as speech
const double _kNoiseFloorMin = -70; // clamp so a truly silent room doesn't make the threshold absurd
const double _kNoiseFloorMax = -20; // clamp so a noisy room doesn't require shouting
const int _kSilenceEndMs = 1000; // sustained silence to consider an utterance finished
const int _kMinSpeechMs = 300; // ignore blips shorter than this
const int _kMaxUtteranceMs = 30000; // safety cap regardless of VAD
// How much audio to keep from just before speech was confirmed, so the very
// first syllable isn't clipped. Deliberately NOT the whole "listening" wait -
// buffering everything since mode was opened sends Whisper a clip that's
// mostly silence, which is a well-known trigger for it to hallucinate and
// loop on repeated phrases (which is what "always shows listening" masked).
const int _kPreRollBytes = _kSampleRate * 2 * 300 ~/ 1000; // 300ms, 16-bit mono

/// Hands-free voice mode: continuously listens, auto-detects when you start
/// and stop talking, transcribes the utterance, streams the assistant's
/// reply as text while speaking it sentence-by-sentence, then goes back to
/// listening - no buttons between turns except to exit.
class VoiceModeOverlay extends StatefulWidget {
  final ApiClient api;
  final String sessionId;
  final bool useKnowledge;
  final void Function(String question, String answer) onTurnComplete;

  const VoiceModeOverlay({
    super.key,
    required this.api,
    required this.sessionId,
    required this.useKnowledge,
    required this.onTurnComplete,
  });

  @override
  State<VoiceModeOverlay> createState() => _VoiceModeOverlayState();
}

class _VoiceModeOverlayState extends State<VoiceModeOverlay> {
  final AudioRecorder _recorder = AudioRecorder();
  final AudioPlayer _audioPlayer = AudioPlayer();

  StreamSubscription<Uint8List>? _pcmSub;
  StreamSubscription<Amplitude>? _amplitudeSub;

  _VoiceState _state = _VoiceState.listening;
  final BytesBuilder _utteranceBuffer = BytesBuilder();
  final List<Uint8List> _preRollChunks = [];
  int _preRollBytes = 0;
  int _speechMsAccumulated = 0;
  int _silenceMsAccumulated = 0;
  bool _everHeardSpeech = false;

  // Adaptive VAD: rather than a fixed dBFS cutoff (which only worked for a
  // loud test signal and never triggered on real, quieter speech), track a
  // rolling low-percentile of recent ambient samples as the noise floor and
  // require the mic level to clear that floor by a margin.
  final List<double> _ambientWindow = [];
  double _noiseFloorDb = -50;
  double _currentAmplitudeDb = -160;

  double get _speechThresholdDb => _noiseFloorDb + _kSpeechMarginDb;

  String _liveTranscript = '';
  String _liveResponse = '';
  String _error = '';

  bool _closing = false;

  @override
  void initState() {
    super.initState();
    _start();
  }

  @override
  void dispose() {
    _closing = true;
    _pcmSub?.cancel();
    _amplitudeSub?.cancel();
    _recorder.dispose();
    _audioPlayer.dispose();
    super.dispose();
  }

  Future<void> _start() async {
    if (!await _recorder.hasPermission()) {
      setState(() => _error = 'Microphone permission denied.');
      return;
    }

    // autoGain/noiseSuppress default to false in record's RecordConfig. Without
    // autoGain, a real mic's natural (quiet) level was never being boosted the
    // way a loud synthetic test WAV effectively already was, so speech rarely
    // crossed a fixed threshold. Enabling it, combined with the adaptive
    // threshold below, is the actual fix for "always shows listening".
    final stream = await _recorder.startStream(const RecordConfig(
      encoder: AudioEncoder.pcm16bits,
      sampleRate: _kSampleRate,
      numChannels: 1,
      autoGain: true,
      noiseSuppress: true,
    ));
    _pcmSub = stream.listen(_onPcmChunk);
    _amplitudeSub = _recorder.onAmplitudeChanged(const Duration(milliseconds: 100)).listen(_onAmplitude);
  }

  void _onPcmChunk(Uint8List chunk) {
    if (_state == _VoiceState.userSpeaking) {
      _utteranceBuffer.add(chunk);
      return;
    }
    if (_state == _VoiceState.listening) {
      // Keep only a short rolling pre-roll while waiting for speech, so an
      // arbitrarily long wait doesn't end up inside the clip sent to Whisper.
      _preRollChunks.add(chunk);
      _preRollBytes += chunk.length;
      while (_preRollBytes > _kPreRollBytes && _preRollChunks.isNotEmpty) {
        _preRollBytes -= _preRollChunks.removeAt(0).length;
      }
    }
  }

  void _onAmplitude(Amplitude amp) {
    // Ignore mic input entirely while the assistant is thinking/talking, or we'd
    // pick up its own speaker output as if it were you (a feedback loop).
    if (_state == _VoiceState.thinking || _state == _VoiceState.aiSpeaking || _state == _VoiceState.transcribing) {
      return;
    }

    const tickMs = 100;

    if (_state == _VoiceState.listening) {
      setState(() {
        _currentAmplitudeDb = amp.current;
        _recordAmbientSample(amp.current);
      });
    } else {
      setState(() => _currentAmplitudeDb = amp.current);
    }

    final isSpeech = amp.current > _speechThresholdDb;

    if (_state == _VoiceState.listening) {
      if (isSpeech) {
        for (final c in _preRollChunks) {
          _utteranceBuffer.add(c);
        }
        _preRollChunks.clear();
        _preRollBytes = 0;
        setState(() {
          _state = _VoiceState.userSpeaking;
          _speechMsAccumulated = tickMs;
          _silenceMsAccumulated = 0;
          _everHeardSpeech = true;
        });
      }
      return;
    }

    if (_state == _VoiceState.userSpeaking) {
      if (isSpeech) {
        _speechMsAccumulated += tickMs;
        _silenceMsAccumulated = 0;
      } else {
        _silenceMsAccumulated += tickMs;
      }

      final longEnoughToCount = _speechMsAccumulated >= _kMinSpeechMs;
      final silenceEndedUtterance = longEnoughToCount && _silenceMsAccumulated >= _kSilenceEndMs;
      final hitMaxDuration = _speechMsAccumulated >= _kMaxUtteranceMs;

      if (silenceEndedUtterance || hitMaxDuration) {
        _finishUtterance();
      }
    }
  }

  /// Feeds one ambient (non-speech-confirmed) amplitude reading into the
  /// rolling noise-floor estimate. Uses a low percentile rather than the
  /// bare minimum so an occasional transient blip (a click, a cough) can't
  /// drag the floor down and make the threshold too sensitive.
  void _recordAmbientSample(double db) {
    _ambientWindow.add(db);
    if (_ambientWindow.length > 20) _ambientWindow.removeAt(0);
    if (_ambientWindow.length < 5) return;
    final sorted = [..._ambientWindow]..sort();
    final p20 = sorted[(sorted.length * 0.2).floor()];
    _noiseFloorDb = p20.clamp(_kNoiseFloorMin, _kNoiseFloorMax);
  }

  Future<void> _finishUtterance() async {
    final pcmBytes = _utteranceBuffer.toBytes();
    _utteranceBuffer.clear();
    _speechMsAccumulated = 0;
    _silenceMsAccumulated = 0;

    if (pcmBytes.isEmpty) {
      setState(() => _state = _VoiceState.listening);
      return;
    }

    setState(() {
      _state = _VoiceState.transcribing;
      _liveTranscript = '';
      _liveResponse = '';
      _error = '';
    });

    try {
      final wav = pcm16ToWav(pcmBytes, sampleRate: _kSampleRate);
      final text = await widget.api.transcribeAudio(wav, 'utterance.wav');
      if (_closing) return;

      if (text.trim().isEmpty) {
        setState(() => _state = _VoiceState.listening);
        return;
      }

      setState(() {
        _liveTranscript = text;
        _state = _VoiceState.thinking;
      });

      await _askAndSpeak(text);
    } catch (e) {
      if (!_closing) setState(() => _error = 'Something went wrong: $e');
    } finally {
      if (!_closing) setState(() => _state = _VoiceState.listening);
    }
  }

  Future<void> _askAndSpeak(String question) async {
    var pendingText = '';
    var fullAnswer = '';
    var switchedToSpeaking = false;

    await for (final event in widget.api.askStream(widget.sessionId, question, widget.useKnowledge)) {
      if (_closing) return;

      final delta = event['delta'] as String?;
      if (delta != null) {
        if (!switchedToSpeaking) {
          switchedToSpeaking = true;
          setState(() => _state = _VoiceState.aiSpeaking);
        }
        fullAnswer += delta;
        pendingText += delta;
        setState(() => _liveResponse = fullAnswer);

        final sentence = _extractCompleteSentence(pendingText);
        if (sentence != null) {
          pendingText = pendingText.substring(sentence.length);
          _enqueueForSpeech(sentence);
        }
      }
    }

    if (pendingText.trim().isNotEmpty) {
      _enqueueForSpeech(pendingText.trim());
    }
    await _drainPlaybackQueue();

    widget.onTurnComplete(question, fullAnswer);
  }

  // Kokoro synthesis takes real wall-clock time (roughly on the order of a
  // sentence's own playback length). Previously each sentence was only
  // synthesized after the PREVIOUS one finished playing, so every sentence
  // boundary had an audible dead gap while waiting on TTS. Fix: start
  // synthesizing a sentence the moment it's extracted from the stream, so
  // synthesis of sentence N+1 overlaps with playback of sentence N - by the
  // time playback catches up, the next clip is usually already sitting in
  // the queue ready to go.
  final List<Future<Uint8List?>> _synthQueue = [];
  bool _draining = false;

  void _enqueueForSpeech(String sentence) {
    _synthQueue.add(widget.api.synthesizeSpeech(sentence).catchError((_) => null));
    unawaited(_drainPlaybackQueue());
  }

  Future<void> _drainPlaybackQueue() async {
    if (_draining) return;
    _draining = true;
    try {
      while (_synthQueue.isNotEmpty && !_closing) {
        final audio = await _synthQueue.removeAt(0);
        if (audio == null || _closing) continue;
        final completer = Completer<void>();
        late final StreamSubscription sub;
        sub = _audioPlayer.onPlayerComplete.listen((_) {
          if (!completer.isCompleted) completer.complete();
          sub.cancel();
        });
        await _audioPlayer.play(BytesSource(audio));
        await completer.future.timeout(const Duration(seconds: 30), onTimeout: () {});
      }
    } finally {
      _draining = false;
    }
  }

  /// Returns the first complete sentence at the start of [text] (including its
  /// trailing punctuation) once enough has streamed in to synthesize it
  /// early, or null if nothing is complete yet.
  String? _extractCompleteSentence(String text) {
    for (var i = 0; i < text.length; i++) {
      final ch = text[i];
      if (ch == '.' || ch == '!' || ch == '?' || ch == '\n') {
        final candidate = text.substring(0, i + 1).trim();
        if (candidate.length > 3) return text.substring(0, i + 1);
      }
    }
    return null;
  }

  Future<void> _close() async {
    _closing = true;
    await _pcmSub?.cancel();
    await _amplitudeSub?.cancel();
    await _recorder.stop();
    await _audioPlayer.stop();
    if (mounted) Navigator.of(context).pop();
  }

  @override
  Widget build(BuildContext context) {
    final c = context.colors;
    return Scaffold(
      backgroundColor: c.bg,
      body: SafeArea(
        child: Column(
          children: [
            Align(
              alignment: Alignment.topRight,
              child: Padding(
                padding: const EdgeInsets.all(16),
                child: IconButton(
                  tooltip: 'Exit voice mode',
                  onPressed: _close,
                  icon: Icon(Icons.close, color: c.fg, size: 28),
                ),
              ),
            ),
            Expanded(
              child: Center(
                child: ConstrainedBox(
                  constraints: const BoxConstraints(maxWidth: 640),
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      _buildOrb(c),
                      const SizedBox(height: 28),
                      Text(_stateLabel, style: TextStyle(color: c.muted, fontSize: 15)),
                      if (_state == _VoiceState.listening || _state == _VoiceState.userSpeaking)
                        Padding(
                          padding: const EdgeInsets.only(top: 6),
                          child: Text(
                            'mic ${_currentAmplitudeDb.toStringAsFixed(0)} dB · threshold ${_speechThresholdDb.toStringAsFixed(0)} dB',
                            style: TextStyle(color: c.muted.withValues(alpha: 0.55), fontSize: 12),
                          ),
                        ),
                      const SizedBox(height: 32),
                      if (_error.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 24),
                          child: Text(_error, style: TextStyle(color: c.danger, fontSize: 13)),
                        ),
                      if (_liveTranscript.isNotEmpty)
                        Padding(
                          padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
                          child: Text(
                            _liveTranscript,
                            textAlign: TextAlign.center,
                            style: TextStyle(color: c.muted, fontSize: 16, fontStyle: FontStyle.italic),
                          ),
                        ),
                      if (_liveResponse.isNotEmpty)
                        Expanded(
                          child: SingleChildScrollView(
                            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
                            child: Text(
                              _liveResponse,
                              textAlign: TextAlign.center,
                              style: TextStyle(color: c.fg, fontSize: 18, height: 1.4),
                            ),
                          ),
                        ),
                    ],
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  String get _stateLabel {
    switch (_state) {
      case _VoiceState.listening:
        return _everHeardSpeech ? 'Listening…' : 'Listening… start talking anytime';
      case _VoiceState.userSpeaking:
        return 'Listening…';
      case _VoiceState.transcribing:
        return 'Got it, one sec…';
      case _VoiceState.thinking:
        return 'Thinking…';
      case _VoiceState.aiSpeaking:
        return 'Speaking…';
    }
  }

  Widget _buildOrb(AppColors c) {
    final active = _state == _VoiceState.userSpeaking || _state == _VoiceState.aiSpeaking;
    final color = _state == _VoiceState.aiSpeaking ? c.accent : c.accent.withValues(alpha: active ? 0.9 : 0.5);

    var size = active ? 140.0 : 110.0;
    if (_state == _VoiceState.listening || _state == _VoiceState.userSpeaking) {
      // React continuously to raw mic level (not just once VAD triggers) so
      // it's immediately obvious whether the mic is picking you up at all.
      final headroom = ((_currentAmplitudeDb - _noiseFloorDb) / 40).clamp(0.0, 1.0);
      size = 100 + headroom * 60;
    }

    return AnimatedContainer(
      duration: const Duration(milliseconds: 100),
      width: size,
      height: size,
      decoration: BoxDecoration(
        shape: BoxShape.circle,
        color: color,
        boxShadow: [
          BoxShadow(color: color.withValues(alpha: 0.4), blurRadius: 40, spreadRadius: 10),
        ],
      ),
      child: (_state == _VoiceState.thinking || _state == _VoiceState.transcribing)
          ? const Padding(
              padding: EdgeInsets.all(40),
              child: CircularProgressIndicator(strokeWidth: 3, color: Colors.white),
            )
          : null,
    );
  }
}
