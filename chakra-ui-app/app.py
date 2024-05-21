from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/api/data')
def get_data():
    # Example data
    return jsonify({"section1": "Data for Section 1", "section2": "Data for Section 2", "section3": "Data for Section 3"})

if __name__ == '__main__':
    app.run(debug=True)
