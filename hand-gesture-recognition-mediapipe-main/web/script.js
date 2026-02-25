import {
  HandLandmarker,
  FilesetResolver
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest";

import * as tf from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.10.0/dist/tf.min.js";
// import * as tflite from "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.9/dist/tf-tflite.esm.js";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const output = document.getElementById("output");

let model;
let handLandmarker;

async function setupCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  video.srcObject = stream;
}

async function init() {
  await setupCamera();

  // model = await tflite.loadTFLiteModel("keypoint_classifier.tflite");
  console.log("Gesture model loaded");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task"
    },
    runningMode: "VIDEO",
    numHands: 1
  });

  console.log("Hand landmarker ready");

  video.onloadeddata = () => detect();
}

async function detect() {
  const results = handLandmarker.detectForVideo(video, performance.now());

  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  if (results.landmarks.length > 0) {
    const landmarks = results.landmarks[0];

    let keypoints = [];
    landmarks.forEach(point => {
      keypoints.push(point.x);
      keypoints.push(point.y);
    });

    const input = tf.tensor([keypoints]);
    const prediction = model.predict(input);
    const data = await prediction.data();

    const index = data.indexOf(Math.max(...data));
    const label = getLabel(index);

    output.innerText = "Gesture: " + label;
  }

  requestAnimationFrame(detect);
}

function getLabel(index) {
  const labels = ["Hello", "Thank You", "I Love You"];
  return labels[index] || "Unknown";
}

init();