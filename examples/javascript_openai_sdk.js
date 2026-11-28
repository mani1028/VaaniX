// Vaani X — OpenAI SDK example (JavaScript / Node.js)
//
// Prerequisites:
//   1. The server is running on localhost:8000 (see README quickstart):
//        telugu-api
//      with TELUGU_API_KEYS set, e.g.:
//        TELUGU_API_KEYS='choose-a-long-random-key'
//   2. Install the SDK:  npm install openai
//   3. Put a real WAV next to this file as clip.wav for the STT part,
//      or delete the transcription section.
//
// What to expect:
//   - speech.wav is written next to this file (24 kHz WAV from the
//     installed Hima voice; model name vaani-tts-piper is an alias for
//     that engine, not Piper — see docs/API_V2.md).
//   - The transcript of clip.wav is printed. Telugu only.
//
// Only baseURL changes vs OpenAI. Everything else is the same code.

import OpenAI from "openai";
import fs from "node:fs";

const client = new OpenAI({
  baseURL: process.env.VAANI_BASE_URL ?? "http://localhost:8000/v1",
  apiKey: (process.env.TELUGU_API_KEYS ?? "choose-a-long-random-key").split(",")[0],
});

// --- Text to speech ---
const speech = await client.audio.speech.create({
  model: "vaani-tts-hima",
  voice: "srinivas", // or "sravani", "telugu_srinivas", "telugu_sravani"
  input: "నమస్కారం",
  response_format: "wav", // wav always works; mp3/opus/flac need ffmpeg
});
fs.writeFileSync("speech.wav", Buffer.from(await speech.arrayBuffer()));
console.log("Wrote speech.wav");

// --- Speech to text ---
const transcription = await client.audio.transcriptions.create({
  model: "vaani-asr-small",
  file: fs.createReadStream("clip.wav"),
  language: "te",
  response_format: "json",
});
console.log("Transcript:", transcription.text);
