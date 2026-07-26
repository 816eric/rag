import 'dart:convert';
import 'dart:typed_data';

/// Wraps raw 16-bit PCM samples in a standard WAV header so the backend
/// (and any browser audio API) can decode them without needing to know the
/// sample format out of band.
Uint8List pcm16ToWav(Uint8List pcmBytes, {required int sampleRate, int numChannels = 1}) {
  final byteRate = sampleRate * numChannels * 2;
  final blockAlign = numChannels * 2;
  final dataLength = pcmBytes.length;

  final header = ByteData(44);
  header.setUint8(0, 0x52); // 'R'
  header.setUint8(1, 0x49); // 'I'
  header.setUint8(2, 0x46); // 'F'
  header.setUint8(3, 0x46); // 'F'
  header.setUint32(4, 36 + dataLength, Endian.little);
  header.buffer.asUint8List().setRange(8, 12, ascii.encode('WAVE'));
  header.buffer.asUint8List().setRange(12, 16, ascii.encode('fmt '));
  header.setUint32(16, 16, Endian.little); // fmt chunk size
  header.setUint16(20, 1, Endian.little); // PCM format
  header.setUint16(22, numChannels, Endian.little);
  header.setUint32(24, sampleRate, Endian.little);
  header.setUint32(28, byteRate, Endian.little);
  header.setUint16(32, blockAlign, Endian.little);
  header.setUint16(34, 16, Endian.little); // bits per sample
  header.buffer.asUint8List().setRange(36, 40, ascii.encode('data'));
  header.setUint32(40, dataLength, Endian.little);

  final result = BytesBuilder();
  result.add(header.buffer.asUint8List());
  result.add(pcmBytes);
  return result.toBytes();
}
