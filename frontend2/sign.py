from flask import Flask, jsonify
import subprocess

app = Flask(__name__)

@app.route('/', methods=['GET'])
def startDetection():
    try:
        # Run final_pred.py and capture the output
        result = subprocess.run(['python', 'D:\\Sign-Lang-to-text&audio-conversion\\Files\\final_pred.py'], capture_output=True, text=True)
        
        # Return a valid JSON response
        return jsonify({"message": "Detection started successfully!", "output": result.stdout})
    except Exception as e:
        # Return error details as JSON
        return jsonify({"message": "Error starting detection", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)