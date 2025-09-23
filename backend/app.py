from flask import Flask, request, jsonify
import base64
import cv2
import datetime
import ipaddress
import math
import numpy as np
import os

import firebase_admin
from firebase_admin import credentials, firestore, storage
from deepface import DeepFace
from zoneinfo import ZoneInfo

app = Flask(__name__)

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "https://csce-4095---it-capstone-i.web.app"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response

# Define the path for credentials
secret_path = '/etc/secrets/firebase_credentials.json'
if os.path.exists(secret_path):
    cred = credentials.Certificate(secret_path)
else:
    cred = credentials.Certificate("backend/firebase/firebase_credentials.json")

# Initialize Firebase app with storage configuration
firebase_admin.initialize_app(cred, {
    "storageBucket": "csce-4095---it-capstone-i.firebasestorage.app"
})
db = firestore.client()
bucket = storage.bucket()  # Initialize storage bucket 

# Timezone for Central Time
CENTRAL_TZ = ZoneInfo("America/Chicago")


def _load_allowed_networks():
    cidr_list = os.getenv(
        "ALLOWED_IP_RANGES",
        "129.120.0.0/16, 129.120.64.0/18, 129.120.96.0/19, 129.120.112.0/20"
    )
    networks = []
    for cidr in cidr_list.split(","):
        cidr = cidr.strip()
        if not cidr:
            continue
        try:
            networks.append(ipaddress.ip_network(cidr, strict=False))
        except ValueError:
            continue
    return networks


ALLOWED_NETWORKS = _load_allowed_networks()
PENDING_MINUTES = int(os.getenv("ATTENDANCE_PENDING_MINUTES", "45"))


def get_client_ip(req):
    header_value = req.headers.get("X-Forwarded-For")
    ip_str = None
    if header_value:
        ip_str = header_value.split(",")[0].strip()
    if not ip_str:
        ip_str = req.remote_addr
    return ip_str or ""


def normalize_ip(ip_str):
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return None
    if isinstance(ip_obj, ipaddress.IPv6Address) and ip_obj.ipv4_mapped:
        ip_obj = ip_obj.ipv4_mapped
    return ip_obj


def is_ip_allowed(ip_str):
    if not ALLOWED_NETWORKS:
        return True
    ip_obj = normalize_ip(ip_str)
    if ip_obj is None:
        return False
    for network in ALLOWED_NETWORKS:
        if ip_obj in network:
            return True
    return False


def parse_time_12h(timestr):
    
    # Parse a 12-hour formatted time string into a datetime.time object.
    timestr = timestr.strip().upper()
    return datetime.datetime.strptime(timestr, "%I:%M%p").time()

def parse_schedule(schedule_str):
    """
    Expects a schedule string like "MWF 8:30AM - 9:50AM"
    The days like MWF are ignored
    Returns start_time and  end_time as datetime.time objects
    """
    parts = schedule_str.strip().split()
    if parts and not any(char.isdigit() for char in parts[0]):
        time_range_str = " ".join(parts[1:])
    else:
        time_range_str = " ".join(parts)
    if "-" not in time_range_str:
        return None, None
    start_str, end_str = time_range_str.split("-", 1)
    start_time = parse_time_12h(start_str)
    end_time = parse_time_12h(end_str)
    return start_time, end_time

def get_attendance_status(now_dt, start_dt, end_dt):
    # Students can start scanning their attendance 5 minutes before class starts
    allowed_start = start_dt - datetime.timedelta(minutes=5)
    # Up to 15 minutes after class start is considered present
    present_cutoff = start_dt + datetime.timedelta(minutes=15)
    
    if now_dt < allowed_start:
        return None, "Attendance cannot be recorded before the allowed time." # If attempted before allowed time
    if now_dt > end_dt:
        return None, "Attendance cannot be recorded after the allowed time." # If attempted after class end time
    if now_dt <= present_cutoff:
        return "Present", None
    else:
        return "Late", None

@app.route("/api/face-recognition", methods=["POST", "OPTIONS"])
def face_recognition():
    if request.method == "OPTIONS":
        return "", 200

    # Temporary filenames for the captured face and the known face downloaded from storage
    temp_captured_path = "temp_captured_face.jpg"
    temp_known_path = "temp_known_face.jpg"

    try:
        client_ip = get_client_ip(request)
        if not is_ip_allowed(client_ip):
            return jsonify({
                "status": "forbidden",
                "message": "Attendance can only be recorded while connected to the UNT EagleNet network."
            }), 403

        data = request.get_json()
        image_b64 = data.get("image")
        class_id = data.get("classId")
        student_id = data.get("studentId")
        
        if not image_b64 or not class_id or not student_id:
            return jsonify({"status": "error", "message": "Missing image, classId, or studentId"}), 400

        # Download the known face image from storage
        # Assumes that known face images are stored under the "known_faces/" folder in our bucket
        blob = bucket.blob(f"known_faces/{student_id}.jpg")
        if not blob.exists():
            return jsonify({"status": "error", "message": "No known face image found for this student."}), 404
        blob.download_to_filename(temp_known_path)

        # Decode the base64 image and save it as the captured face
        image_data = base64.b64decode(image_b64.split(',')[1])
        np_arr = np.frombuffer(image_data, np.uint8)
        captured_img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if captured_img is None:
            return jsonify({"status": "error", "message": "Captured image could not be decoded."}), 400
        cv2.imwrite(temp_captured_path, captured_img)

        # Use DeepFace to verify the face. Compare the captured face with the known face downloaded from storage
        """
        print("Running DeepFace.verify...")
        verify_result = DeepFace.verify(
        img1_path=temp_captured_path,
        img2_path=temp_known_path,
        model_name="VGG-Face",
        enforce_detection=False
        )
        print("DeepFace.verify completed.")
        """
        # Verified result for testing purposes
        # Uncomment the above DeepFace.verify code and comment the below lines for real scan
        verify_result = {
        "verified": True,
        "distance": 0.12,
        "max_threshold_to_verify": 0.3
        }

        print("DeepFace verify result:", verify_result)
        if not verify_result.get("verified", False):
            return jsonify({"status": "fail", "message": "Face not recognized"}), 404

        # Get current central time
        now_central = datetime.datetime.now(CENTRAL_TZ)
        today_str = now_central.strftime("%Y-%m-%d")
        doc_id = f"{class_id}_{student_id}_{today_str}"

        # Check if attendance record already exists
        attendance_doc = db.collection("attendance").document(doc_id).get()
        if attendance_doc.exists:
            existing_data = attendance_doc.to_dict()
            if existing_data.get("status") == "Pending":
                pending_until = existing_data.get("pendingExpiresAt")
                if isinstance(pending_until, datetime.datetime):
                    remaining_seconds = max((pending_until - now_central).total_seconds(), 0)
                else:
                    remaining_seconds = PENDING_MINUTES * 60
                return jsonify({
                    "status": "pending",
                    "message": "Attendance is already pending final confirmation.",
                    "attendance_status": existing_data.get("pendingFinalStatus"),
                    "pending_minutes": math.ceil(remaining_seconds / 60),
                    "pending_seconds": int(remaining_seconds),
                    "doc_id": doc_id,
                }), 200
            return jsonify({"status": "already_marked", "message": "Attendance already recorded today."}), 200

        # Retrieve class document to fetch the class schedule
        class_doc = db.collection("classes").document(class_id).get()
        if not class_doc.exists:
            return jsonify({"status": "error", "message": "Class not found"}), 404
        class_data = class_doc.to_dict()
        schedule_str = class_data.get("schedule", "").strip()
        if not schedule_str:
            return jsonify({"status": "error", "message": "No schedule defined for this class"}), 400

        start_time, end_time = parse_schedule(schedule_str)
        if not start_time or not end_time:
            return jsonify({"status": "error", "message": "Invalid schedule format"}), 400

        start_dt = datetime.datetime(
            now_central.year, now_central.month, now_central.day,
            start_time.hour, start_time.minute, 0, 0,
            tzinfo=CENTRAL_TZ
        )
        end_dt = datetime.datetime(
            now_central.year, now_central.month, now_central.day,
            end_time.hour, end_time.minute, 0, 0,
            tzinfo=CENTRAL_TZ
        )

        status, error_msg = get_attendance_status(now_central, start_dt, end_dt)
        if error_msg:
            return jsonify({"status": "fail", "message": error_msg}), 400

        print("Computed attendance status:", status)
        # Create attendance record document in Firebase
        pending_expires_at = now_central + datetime.timedelta(minutes=PENDING_MINUTES)
        attendance_record = {
            "studentID": student_id,
            "classID": class_id,
            "date": now_central,
            "status": "Pending",
            "pendingFinalStatus": status,
            "pendingStartedAt": now_central,
            "pendingExpiresAt": pending_expires_at,
            "pendingDurationMinutes": PENDING_MINUTES,
            "initialCheckIp": client_ip,
            "verificationMetrics": {
                "distance": verify_result.get("distance"),
                "threshold": verify_result.get("max_threshold_to_verify"),
            },
        }
        db.collection("attendance").document(doc_id).set(attendance_record)
        # Data from successful scan
        return jsonify({
            "status": "pending",
            "recognized_student": student_id,
            "attendance_status": status,
            "distance": verify_result.get("distance"),
            "threshold": verify_result.get("max_threshold_to_verify"),
            "pending_minutes": PENDING_MINUTES,
            "pending_seconds": PENDING_MINUTES * 60,
            "doc_id": doc_id,
            "message": "Attendance recorded as pending. Keep the application open for 45 minutes so we can confirm you remain on EagleNet.",
        }), 200

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

    finally:
        # Clean up temporary files that were created
        if os.path.exists(temp_captured_path):
            os.remove(temp_captured_path)
        if os.path.exists(temp_known_path):
            os.remove(temp_known_path)


@app.route("/api/attendance/finalize", methods=["POST", "OPTIONS"])
def finalize_attendance():
    if request.method == "OPTIONS":
        return "", 200

    try:
        client_ip = get_client_ip(request)
        data = request.get_json() or {}
        class_id = data.get("classId")
        student_id = data.get("studentId")

        if not class_id or not student_id:
            return jsonify({
                "status": "error",
                "message": "Missing classId or studentId"
            }), 400

        now_central = datetime.datetime.now(CENTRAL_TZ)
        today_str = now_central.strftime("%Y-%m-%d")
        doc_id = f"{class_id}_{student_id}_{today_str}"
        attendance_ref = db.collection("attendance").document(doc_id)
        attendance_doc = attendance_ref.get()

        if not attendance_doc.exists:
            return jsonify({
                "status": "error",
                "message": "No pending attendance record found for this class and student."
            }), 404

        record = attendance_doc.to_dict()
        if record.get("status") != "Pending":
            return jsonify({
                "status": "error",
                "message": "Attendance is already finalized.",
                "current_status": record.get("status")
            }), 400

        pending_until = record.get("pendingExpiresAt")
        if isinstance(pending_until, datetime.datetime) and now_central < pending_until:
            seconds_remaining = max((pending_until - now_central).total_seconds(), 0)
            return jsonify({
                "status": "pending",
                "message": "Attendance finalization is not ready yet.",
                "pending_seconds": int(seconds_remaining),
                "pending_minutes": math.ceil(seconds_remaining / 60),
            }), 200

        if not is_ip_allowed(client_ip):
            attendance_ref.update({
                "status": "Rejected",
                "finalizedAt": now_central,
                "finalizedIp": client_ip,
                "rejectionReason": "Client not connected to approved network during final confirmation.",
            })
            return jsonify({
                "status": "denied",
                "message": "Attendance could not be finalized because you are no longer on the EagleNet network.",
            }), 403

        final_status = record.get("pendingFinalStatus", "Present")
        attendance_ref.update({
            "status": final_status,
            "finalizedAt": now_central,
            "finalizedIp": client_ip,
        })
        return jsonify({
            "status": "success",
            "final_status": final_status,
            "message": "Attendance finalized successfully."
        }), 200

    except Exception as exc:
        return jsonify({"status": "error", "message": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
