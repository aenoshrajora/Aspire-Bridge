from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/')
def career():
    return render_template("hometest.html")

@app.route('/predict', methods=['POST', 'GET'])
def result():
    if request.method == 'POST':
        try:
            # Get input from the form
            result = request.form
            print(result)
            
            # Convert form data to dictionary and extract values
            res = result.to_dict(flat=True)
            print("res:", res)
            
            arr1 = res.values()
            arr = [value for value in arr1]
            
            # Convert input data to numeric format
            data = np.array(arr, dtype=float)
            data = data.reshape(1, -1)  # Reshape for prediction
            print(data)
            
            # Load the trained model
            loaded_model = pickle.load(open("careerlast.pkl", 'rb'))
            
            # Make predictions
            predictions = loaded_model.predict(data)
            print(predictions)
            
            # Get probabilities for additional recommendations
            pred = loaded_model.predict_proba(data)
            print(pred)
            
            pred = pred > 0.05  # Filter probabilities above a threshold
            
            # Determine other job matches
            i, j, index = 0, 0, 0
            res = {}
            final_res = {}
            
            while j < 17:  # Assuming 17 job categories
                if pred[i, j]:
                    res[index] = j
                    index += 1
                j += 1
            
            index = 0
            for key, values in res.items():
                if values != predictions[0]:
                    final_res[index] = values
                    print('final_res[index]:', final_res[index])
                    index += 1
            
            # Job categories dictionary
            jobs_dict = {
                0: 'AI ML Specialist',
                1: 'API Integration Specialist',
                2: 'Application Support Engineer',
                3: 'Business Analyst',
                4: 'Customer Service Executive',
                5: 'Cyber Security Specialist',
                6: 'Data Scientist',
                7: 'Database Administrator',
                8: 'Graphics Designer',
                9: 'Hardware Engineer',
                10: 'Helpdesk Engineer',
                11: 'Information Security Specialist',
                12: 'Networking Engineer',
                13: 'Project Manager',
                14: 'Software Developer',
                15: 'Software Tester',
                16: 'Technical Writer'
            }
            
            # Get the predicted job
            data1 = predictions[0]
            print(data1)
            
            return render_template(
                "testafter.html", 
                final_res=final_res, 
                job_dict=jobs_dict, 
                job0=data1
            )
        except ValueError as e:
            print(f"ValueError: {e}")
            return "Error: Please provide valid numeric inputs."
        except Exception as e:
            print(f"Exception: {e}")
            return "An unexpected error occurred. Please try again."
            
if __name__ == '__main__':
    app.run(debug=True)
    