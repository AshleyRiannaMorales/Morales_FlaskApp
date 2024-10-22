from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the KNN model
knn_model = joblib.load('model/knn_model.pkl')
nb_model = joblib.load('model/naive_bayes_model.pkl')
dtree_model = joblib.load('model/dtree_model.pkl')


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Retrieve data from form
        clump_thickness = int(request.form['clump_thickness'])
        uniformity_cellsize = int(request.form['uniformity_cellsize'])
        uniformity_cellshape = int(request.form['uniformity_cellshape'])
        marginal_adhesion = int(request.form['marginal_adhesion'])
        epithelial_cellsize = int(request.form['epithelial_cellsize'])
        bland_chromatin = int(request.form['bland_chromatin'])
        normal_nucleoli = int(request.form['normal_nucleoli'])
        mitoses = int(request.form['mitoses'])
        classifier = request.form['classifier']

        # Prepare input for prediction
        input_data = np.array([[clump_thickness, uniformity_cellsize, uniformity_cellshape,
                                marginal_adhesion, epithelial_cellsize,
                                bland_chromatin, normal_nucleoli, mitoses]])
        
        # Choose the appropriate model based on the selected classifier
        if classifier == "Nearest Neighbor":
            prediction = knn_model.predict(input_data)
        elif classifier == "Naive Bayes":
            prediction = nb_model.predict(input_data)
        elif classifier == "Decision Tree":
            prediction = dtree_model.predict(input_data)
        else:
            prediction = ["Unknown"]

        # Format prediction result
        prediction_result = ''
        if prediction[0] == 2:
            prediction_result = "Benign"
        elif prediction[0] == 4:
            prediction_result = "Malignant"

        # Return result to the template
        return render_template('index.html',
                               inputs=f"{clump_thickness}, {uniformity_cellsize}, {uniformity_cellshape}, "
                                      f"{marginal_adhesion}, {epithelial_cellsize}, "
                                      f"{bland_chromatin}, {normal_nucleoli}, {mitoses}",
                               classifier=classifier,
                               prediction=prediction_result)

    return render_template('index.html')
    
if __name__ == '__main__':
    app.run(debug=True)
