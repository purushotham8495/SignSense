from flask import Flask, render_template, Response, jsonify, request
import cv2
from sign_processor import SignProcessor

app = Flask(__name__)
processor = SignProcessor()
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            try:
                processed_frame = processor.process_frame(frame)
            except Exception as e:
                print("Error processing frame:", e)
                processed_frame = frame

            ret, buffer = cv2.imencode('.jpg', processed_frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/state')
def get_state():
    return jsonify({
        "current_symbol": processor.current_symbol,
        "sentence": processor.str,
        "word1": processor.word1,
        "word2": processor.word2,
        "word3": processor.word3,
        "word4": processor.word4,
        "is_capturing": processor.capture_mode,
        "is_capturing": processor.capture_mode,
        "capture_count": processor.capture_count,
        "is_active": processor.is_active
    })

@app.route('/action', methods=['POST'])
def handle_action():
    action = request.json.get('action')
    try:
        if action == 'action1':
            processor.action1()
        elif action == 'action2':
            processor.action2()
        elif action == 'action3':
            processor.action3()
        elif action == 'action4':
            processor.action4()
        elif action == 'clear':
            processor.clear()
        elif action == 'speak':
            processor.speak_text()
        elif action == 'backspace':
            processor.backspace()
        elif action == 'space':
            processor.space()
        return jsonify({"status": "success"})
    except Exception as e:
        print("Action error:", e)
        return jsonify({"status": "error"})

@app.route('/toggle_active', methods=['POST'])
def toggle_active():
    val = request.json.get('active')
    processor.toggle_active(val)
    return jsonify({"status": "updated"})

@app.route('/start_capture', methods=['POST'])
def start_capture():
    word = request.json.get('word')
    if word:
        processor.start_capture(word)
        return jsonify({"status": "started"})
    return jsonify({"status": "Error"})

@app.route('/cancel_capture', methods=['POST'])
def cancel_capture():
    processor.cancel_capture()
    return jsonify({"status": "cancelled"})

@app.route('/add_word', methods=['POST'])
def add_word():
    new_word = request.json.get('word')
    if new_word:
        processor.add_custom_word(new_word)
        return jsonify({"status": "Added"})
    return jsonify({"status": "Error"})


if __name__ == "__main__":
    app.run(debug=True, threaded=True, host='0.0.0.0', port=5000)
