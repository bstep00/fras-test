import React, { useState, useRef, useEffect } from "react";
import { useNavigate } from "react-router-dom";

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL ||
  "https://facialrecognitionattendancesystem.onrender.com";

const FaceScanner = ({ selectedClass, studentId }) => {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const finalizeTimeoutRef = useRef(null);
  const [scanning, setScanning] = useState(false);
  const [captureDisabled, setCaptureDisabled] = useState(false);
  const [statusNotice, setStatusNotice] = useState(null);
  const [pendingInfo, setPendingInfo] = useState(null);
  const navigate = useNavigate();

  const startVideo = () => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        streamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((error) => {
        console.error("Error accessing camera:", error);
        setStatusNotice("❌ Unable to access the camera.");
      });
  };

  const stopVideo = () => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const clearFinalizeTimer = () => {
    if (finalizeTimeoutRef.current) {
      clearTimeout(finalizeTimeoutRef.current);
      finalizeTimeoutRef.current = null;
    }
  };

  useEffect(() => {
    startVideo();
    return () => {
      stopVideo();
      clearFinalizeTimer();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const scheduleFinalization = (seconds) => {
    clearFinalizeTimer();
    const numericSeconds = Number(seconds);
    const safeSeconds = Math.max(Number.isFinite(numericSeconds) ? numericSeconds : 0, 0);
    finalizeTimeoutRef.current = window.setTimeout(() => {
      finalizeAttendance(selectedClass, studentId);
    }, safeSeconds * 1000);
  };

  const finalizeAttendance = async (classId, studentIdParam) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/attendance/finalize`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ classId, studentId: studentIdParam }),
      });

      const data = await response.json();
      console.log("Finalize result:", data);

      if (data.status === "success") {
        clearFinalizeTimer();
        setStatusNotice(`✅ Attendance finalized! Status: ${data.final_status}.`);
        setPendingInfo(null);
        setCaptureDisabled(false);
        navigate(`/student/classes/${classId}`, { replace: true });
      } else if (data.status === "denied") {
        clearFinalizeTimer();
        setStatusNotice(`❌ ${data.message}`);
        setPendingInfo(null);
        setCaptureDisabled(false);
      } else if (data.status === "pending") {
        const retrySeconds = Math.max(data.pending_seconds ?? 60, 30);
        setStatusNotice(
          "⏳ We're still waiting to confirm your connection. Please keep the app open."
        );
        setPendingInfo((current) => {
          const finalizeAt = new Date(Date.now() + retrySeconds * 1000);
          if (!current) {
            return {
              classId,
              studentId: studentIdParam,
              pendingSeconds: retrySeconds,
              finalizeAt,
            };
          }
          return {
            ...current,
            pendingSeconds: retrySeconds,
            finalizeAt,
          };
        });
        scheduleFinalization(retrySeconds);
      } else {
        clearFinalizeTimer();
        setStatusNotice(`❌ ${data.message || "Unable to finalize attendance."}`);
        setPendingInfo(null);
        setCaptureDisabled(false);
      }
    } catch (error) {
      console.error("Error finalizing attendance:", error);
      setStatusNotice(
        "❌ Error finalizing attendance. We will try again in a minute if you remain on EagleNet."
      );
      scheduleFinalization(60);
    }
  };

  const capturePhoto = async () => {
    if (!videoRef.current) return;

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    canvas.getContext("2d").drawImage(videoRef.current, 0, 0);
    const dataURL = canvas.toDataURL("image/jpeg");

    try {
      setScanning(true);
      setStatusNotice(null);

      const response = await fetch(`${API_BASE_URL}/api/face-recognition`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: dataURL, classId: selectedClass, studentId }),
      });

      const result = await response.json();
      console.log("Recognition result:", result);

      if (result.status === "pending") {
        const pendingSeconds = Math.max(
          result.pending_seconds ?? (result.pending_minutes ?? 45) * 60,
          0
        );
        const minutesText = Math.max(Math.ceil(pendingSeconds / 60), 1);
        const finalizeAt = new Date(Date.now() + pendingSeconds * 1000);
        setPendingInfo({
          classId: selectedClass,
          studentId,
          pendingSeconds,
          finalizeAt,
        });
        setStatusNotice(
          `✅ Attendance marked as ${result.attendance_status}, pending confirmation. Keep this page open for ${minutesText} minute${
            minutesText === 1 ? "" : "s"
          } so we can verify you remain on EagleNet.`
        );
        setCaptureDisabled(true);
        stopVideo();
        scheduleFinalization(pendingSeconds);
      } else if (result.status === "already_marked") {
        setStatusNotice("⚠️ Attendance already recorded today.");
        navigate(`/student/classes/${selectedClass}`);
      } else if (result.status === "forbidden") {
        setStatusNotice(
          result.message || "❌ Attendance must be recorded from the EagleNet network."
        );
      } else {
        setStatusNotice(`❌ ${result.message || "Attendance could not be recorded."}`);
      }
    } catch (error) {
      console.error("Error during recognition:", error);
      setStatusNotice("❌ Error during face recognition.");
    } finally {
      setScanning(false);
    }
  };

  return (
    <div>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        style={{ width: "300px", height: "300px", background: "#000" }}
      ></video>
      <button
        onClick={capturePhoto}
        disabled={scanning || captureDisabled}
        className="bg-green-600 text-white px-3 py-1 rounded mt-4 disabled:opacity-60 disabled:cursor-not-allowed"
      >
        {scanning ? "Scanning..." : "Capture Face"}
      </button>
      {statusNotice && (
        <p className="mt-4 text-sm text-gray-800 whitespace-pre-line">{statusNotice}</p>
      )}
      {pendingInfo && pendingInfo.finalizeAt && (
        <p className="mt-2 text-xs text-gray-600">
          We'll re-check your connection around {pendingInfo.finalizeAt.toLocaleTimeString()}. Please
          keep this page open and stay on EagleNet.
        </p>
      )}
    </div>
  );
};

export default FaceScanner;
