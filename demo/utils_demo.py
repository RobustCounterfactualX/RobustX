import time
from sklearn.model_selection import train_test_split
import streamlit as st
import streamlit as st
from robustx.datasets.ExampleDatasets import get_example_dataset
from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel
from robustx.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from demo_code import *
from robustx.lib.tasks.ClassificationTask import ClassificationTask


def get_intro():
    st.markdown(
    "<h1 style='font-size: 30px;'>RobustX: Robust Counterfactual Explanations Made Easy</h1>", 
    unsafe_allow_html=True)
    st.markdown(
    "<h2 style='font-size: 24px; font-weight: normal;'>Junqi Jiang, Luca Marzari, Aaryan Purohit and Francesco Leofante</h2>", 
    unsafe_allow_html=True
    )
    st.markdown(
    "<h2 style='font-size: 15px; margin-top: -25px; font-weight: normal;'><em>luca.marzari@univr.it, {junqi.jiang,aaryan.purohit22,f.leofante}@imperial.ac.uk</em></h2>", 
    unsafe_allow_html=True
    )
    # Divider with reduced margin and padding
    st.markdown(
        "<hr style='border: 1px solid black; margin-top: -10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True)
  

    _, col2, _ = st.columns([1,2,1])  # Middle column is wider
    with col2:
        st.image("images/overview_new.png", use_container_width=True, caption="***RobustX*** overview")
    
    st.write("The increasing use of machine learning models to aid decision-making in high-stakes industries like finance and healthcare demands explainability to build trust. Counterfactual Explanations (CEs) provide valuable insights into model predictions by showing how slight changes in input data could lead to different outcomes. A key aspect of CEs is their robustness, which ensures that the desired outcomes remain stable even with minor alterations to the input. Robustness is important since produced CEs should hold up in the future should the original model be altered or replaced. Despite the importance of robustness, there has been a lack of standardised tools to comprehensively evaluate and compare robust CE generation methods. To address this, *****RobustX***** was developed as an open-source Python library aimed at benchmarking the robustness of various CE methods. ***RobustX*** provides a systematic framework for generating, evaluating, and comparing CEs with a focus on robustness, enabling fair and effective benchmarking. The library is highly extensible, allowing for custom models, datasets, and tools to be integrated, making it an essential tool for enhancing the reliability and interpretability of machine learning models in critical applications.")
    

    st.write("### Case study: Loan application scenario")


    st.write("Consider a scenario where a deep neural network decides whether a loan application is accepted or denied...")
    st.image("images/motivation.png", use_container_width=True, caption="Motivation for ***RobustX***: a loan application scenario where a deep neural network decides whether to accept or deny the application. A counterfactual explanation can explain how the applicant should change their application to get the loan approved...")

    st.write("***RobustX*** provides a systematic framework for generating, evaluating, and comparing counterfactual explanations with a focus on robustness. The library is highly extensible, allowing for custom models, datasets, and tools to be integrated, making it an essential tool for enhancing the reliability and interpretability of machine learning models in critical applications.")

    st.image("images/example_CE.png", use_container_width=True, caption="Example of a counterfactual explanation. The counterfactual explanation shows how the applicant should change their application to get the loan approved.")

    st.write("### This demo: ***RobustX*** to generate counterfactual explanations robust against model changes")

    st.write("We are interested in generating a counterfactual explanation. Let us consider the following problem...")
    
    # In literature there exist other toolboxes to generate counterfactual explanations. For instance, CARLA, provides a set of methods to generate explanations for a given input instance. Nonetheless, these toolboxes do not take into account the robustness of the generated explanations against different type of perturbations. ***RobustX*** fill this gap providing a self-contained python library with a set of methods to generate counterfactual explanations and evaluate their robustness. For instace, let us consider the following problem...")

    st.image("images/CFX_example_rob.png", use_container_width=True, caption="Vignette illustrating the problem of robustness under model changes. A counterfactual explanation is initially generated for a trained model (left). Then, the model is updated to include new data (right). This step might induce slight changes in the decision boundary of the model, ultimately invalidating the counterfactual explanation generated in the first step.")

    st.write("Let's see this in practice with ***RobustX***...")


def get_task_part():

    st.markdown(
        "<hr style='border: 1px solid black; margin-top: -10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True)
    st.write("### Step 1: Defining the Task")

    _, col2, _ = st.columns([1,2,1])  # Middle column is wider
    with col2:
        st.image("images/task.png", use_container_width=True, caption="How to generate a Task instance?")


    st.write("To generate a Task instance, you need to define the following:")
    st.write("1. **BaseModel**: the class that will contain the model to be trained/explained.\n2. **DatasetLoader**: the class that will contain the dataset on which training/is trained the model.")

    st.write("***RobustX*** provides a set of models and datasets that can be used out-of-the-box. You can also define your custom models and datasets to be used in the library. For this demo, we will create a random PyTorch model and use a custom dataset, composed by only two features and one output.")

    st.write("#### Define the model")
    st.write("***RobustX*** provides simple way to define a model using a set of known frameworks in literature e.g., Pytorch, Keras, Scikit-Learn. You can define a model by inheriting from the BaseModel class and implementing the forward method.")
    st.write("For simplicity, let us consider a Neural Network with two input and one output (class 0- loan denied and class 1- loan accepted)")

    st.code("""
            from robustx.lib.models.pytorch_models.SimpleNNModel import SimpleNNModel

            # let's define a simple pytorch model with 2 input features, 2 hidden layer of 8 nodes activated with ReLU and 1 sigmoid output node. We can also set the seed for reproducibility purposes.
            model = SimpleNNModel(2, [8, 8], 1, seed=0)
            
            # to check the model architecture you can use the following code
            print(model)
            """, language="python", line_numbers=True, wrap_lines=True)
    
    model = SimpleNNModel(2, [8, 8], 1, seed=0)
    st.write("Here we report the model architecture:")
    st.code(str(model.get_torch_model), language="python")

    st.write("#### Define the dataset")

    st.write("Regarding the dataset, you can decide to load a custom dataset or use one of the datasets provided by ***RobustX***.")
    #  In this latter case, you decide between:\n - *Adult dataset*\n - *Ionosphere dataset*\n - *Iris dataset*\n - *Titanic dataset*.\n and load them using the following code")


    st.code("""
    import pandas as pd
    from robustx.datasets.ExampleDatasets import get_example_dataset

    # Create an instance of CsvDatasetLoader which will load the dataset.
    ionosphere_loader = get_example_dataset("ionosphere")

    # Load the dataset
    ionosphere_loader.load_data()
            
    # Apply default preprocessing
    ionosphere_loader.default_preprocess()

    # Display the first 5 rows of the dataset
    print(ionosphere_loader.data.head())
            
    """, language="python", line_numbers=True)

    ionosphere_loader = get_example_dataset("ionosphere")

    # Load the dataset
    ionosphere_loader.load_data()

    # Apply default preprocessing
    ionosphere_loader.default_preprocess()

    # Display the first 5 rows of the data
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(ionosphere_loader.data.head(), width=500)

    st.write("For this demo, we will use a custom dataset composed by only two features and one output we provide for you. ***RobustX*** provides a class called CsvDatasetLoader that allows you to load a custum dataset from a CSV file. Here is an example:")             


    st.code("""
    import pandas as pd
    from robustx.datasets.custom_datasets.CsvDatasetLoader import CsvDatasetLoader

    # Load the dataset
    csv_path = "test.csv"  # Path to the CSV file
    target_column = "target"  # Name of the target column

    # Create an instance of CsvDatasetLoader which will load the dataset.
    dataset_loader = CsvDatasetLoader(csv=csv_path, target_column=target_column)
    
    # to access and print the dataset in as a pd Dataframe you can use the following code
    print(dataset_loader.data)
        
    """, language="python", line_numbers=True)

    
    csv_path = "dataset/custom_dataset.csv"  # Path to the CSV file
    target_column = "target"  # Name of the target column

    # Create an instance of CsvDatasetLoader
    dataset_loader = CsvDatasetLoader(csv=csv_path, target_column=target_column)
  
    st.write("Here we report the synthetic dataset loaded:")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.dataframe(dataset_loader.data.head(), width=500)

    st.write("Scatter plot of the synthetic dataset colored by target...")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=dataset_loader.data, x=dataset_loader.data["feature1"], y=dataset_loader.data["feature2"], hue='target', palette='coolwarm', s=100)
        plt.title('Synthetic Dataset Scatter Plot')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')

        # Show the plot in Streamlit
        st.pyplot(plt)


    st.write("#### Training the model on the dataset (optional)")

    st.write("Now that we have defined the model and the dataset, we can train the model on the dataset. This step is optional, as you can also load a pre-trained model. To train the model, we first perform a split of the dataset in training and testing data (80-20):")


    # create a train-test split
    X_train, X_test, y_train, y_test = train_test_split(dataset_loader.data.drop(columns=[target_column]), dataset_loader.data[target_column], test_size=0.2, random_state=0)

    training_done = False
 
    accuracy = model.compute_accuracy(X_test.values, y_test.values)
    st.write(f"Model Accuracy before training: {accuracy * 100:.1f}%")

    if not training_done:
        if st.button('Train Neural Network'):
            with st.spinner("Training the neural network..."):
                model.train(X_train, y_train, epochs=100)
                time.sleep(1)

        
            accuracy = model.compute_accuracy(X_test.values, y_test.values)
            st.write(f"Training completed! Model Accuracy: {accuracy * 100:.1f}%")
            training_done = True

            # Display results
            st.write("We can now visualize the decision boundary of the trained model in the test dataset...")

           
            # Create columns to center the plot
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                name_file = "decision_boundary"
                st.image(f"images/{name_file}.png", use_container_width=True)


            st.write("We can now generate a Task instance using the model and the dataset...")
            st.code("""
            from robustx.lib.tasks.ClassificationTask import ClassificationTask

            task = ClassificationTask(model, dataset_loader)
                
            """, language="python", line_numbers=True)

            
            st.write("Task instance created!")

            st.write("Consider the following input instance for which the model correctly predicts the class 0 (i.e., loan denied):")

            st.code("""
                # get the first input instance for which the model predicts class 0
                input_instance = X_test[y_test == 0].iloc[0]
                print("Input_instance: ", input_instance.values)
                >> [-2.17259632  1.4679443]

                model_output = model.predict(torch.tensor(input_instance.values, dtype=torch.float32).reshape(1, -1))
                print("Propagating input_instance through the model:", model_output.values)
                >> [[0.21322772]]
                print("The model's prediction class for the input is: ", model.predict_single(input_instance))
                >> 0
            """, language="python", line_numbers=True)



            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(f"images/input_instance.png", use_container_width=True)

            

            st.write("Let's see how to use ***RobustX*** to generate a robust counterfactual explanation against model changes...")
    
         


def get_generation_part():

    st.markdown(
        "<hr style='border: 1px solid black; margin-top: -10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True)
    st.write("### Step 2: Generating Counterfactual Explanations using ***RobustX***")

    _, col2, _ = st.columns([1,2,1])  # Middle column is wider
    with col2:
        st.image("images/generation.png", use_container_width=True, caption="How to generate a Counterfactual Exaplanation (CE) with ***RobustX***?")


    st.write("***RobustX*** provides a predefined set of methods to generate counterfactual explanations. You can also define your custom methods to be used in the library. In the following we generate CEs using some robust, as well as some non-robust methods.")
    
    
    # For the sake of simplicity, we report here few methods not-robust (which can also be easily found in other libraries as well)  and a small set of robust methods, peculiar to this library. Hence, we show below the importance of having these when considering the generation of counterfactual explanations.")
   
    # Sample data for your table
    data = {
        "Method Name": ["BinaryLinearSearch", "KDTreeNNCE", "MCE", "NNCE", "Wachter", "RNCE", "MCER", "STCE", "APAS"],
        "Extended Name": [
            "Binary Linear search for Counterfactual Explanation", 
            "Nearest neighbours with KD tree Counterfactual Explanation", 
            "Mixed Integer Linear Programming (MILP) Counterfactual Explanation", 
            "Nearest Neighbours Counterfactual Explanation (NNCE)", 
            "Wachter's method for finding counterfactual explanations",
            "NNCE with provable Δ-Robustness evaluation",
            "MILP Counterfactual Explanation with provable Δ-Robustness evaluation",
            "T-Rex method for finding robust counterfactual explanations.",
            "APΔS method for finding probabilistically Δ robust counterfactual explanations."
        ],
        "Robust": ['❌', '❌', '❌', '❌', '❌', '✅', '✅', '✅', '✅'],
        }


    # Create a DataFrame
    df = pd.DataFrame(data)
    # Show the table (optional)
    st.dataframe(df)

   
    st.write("Here below we show how easy is to generate a counterfactual explanation using ***RobustX***. For instance, we can generate a counterfactual explanation using the MCE method as follows:")

    st.code("""
            from robustx.lib.generators.MCE import MCE

            # instantiate the MCE generator
            generator_mce = MCE(task)
            ce = generator_mce._generation_method(input_instance, neg_value=0)
            print("MCE CE is: ", ce.values)
            >> [[ 1.34792239 -0.976032]]

            model_output = model.predict(torch.tensor(ce.values, dtype=torch.float32).reshape(1, -1))
            print("propagating CE through the model:", model_output.values)
            >> [[0.500025]]

            print("The model's prediction class for the input is: ", model.predict_single(ce))
            >> 1
            """, language="python", line_numbers=True)


    st.write("Suppose we select all not-robust method and only one robust approach to generate the counterfactual. Let's see how to generate the counterfactual explanations...")


    # Create a list to store selected methods
    selected_methods = []

    # Show the table with checkboxes for selection
    for index, row in df.iterrows():
        method_name = row["Method Name"]
        selected = st.checkbox(f"Select {method_name}", key=method_name)
        
        # If selected, append to selected_methods list
        if selected:
            selected_methods.append(method_name)

    # Display the selected methods
    st.write("### Selected Methods:")
    st.write(selected_methods)

   
    if st.button('Generate Counterfactual Explanations'):
        with st.spinner("Generating counterfactual explanations..."):
            time.sleep(1)
            st.write("Counterfactual explanations generated!")
            st.write("We can now visualize the counterfactual explanations...")
            # Create columns to center the plot
            col1, col2, col3 = st.columns([1, 2, 1])
            name_file=f"decision_boundary_with_ces_"
            for i, method in enumerate(selected_methods):
                if i == len(selected_methods) - 1:
                    name_file += method
                else:
                    name_file += method + "_"
            print(name_file)
            with col2:
                st.image(f"images/{name_file}_before_retraining.png", use_container_width=True)


        st.write("As we can notice, all the CEs generated correctly flip the class of the input instance from 0 to 1. Importantly, we can notice that the robust method (RNCE) generates the most distant CE from the decision boundary. This is important as it ensures that the CE will be robust against small changes in the model. Let's see what happens when we retrain the model with new data...")
        # Divider with reduced margin and padding
       



def get_retraining_part():

    st.markdown(
        "<hr style='border: 1px solid black; margin-top: -10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True)
    st.write("### Step 3: Retraining the Model")

    X_new, y_new = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,class_sep=2.0, flip_y=0, random_state=2001)
    X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, y_new, test_size=0.2, random_state=2001)

    df_new = pd.DataFrame(X_new, columns=['feature1', 'feature2'])
    df_new['target'] = y_new
    df_new['is_new'] = 'New'
    df = pd.read_csv("dataset/custom_dataset.csv")
    df['is_new'] = 'Original'
    df_combined = pd.concat([df, df_new], ignore_index=True)

    st.write("Scatter plot of the new synthetic dataset colored by label...")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_combined[df_combined['is_new'] == 'Original'], x='feature1', y='feature2', hue='target', palette='coolwarm', s=200)
    sns.scatterplot(data=df_combined[df_combined['is_new'] == 'New'], x='feature1', y='feature2', hue='target', palette='coolwarm', s=200, marker='X', edgecolor='black', linewidth=1.5)

    plt.title('Synthetic Dataset Scatter Plot')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

    # Show the plot in Streamlit
    _, col2, _ = st.columns([1,2,1])  # Middle column is wider
    with col2:
        st.pyplot(plt)

    st.write("We can now retrain the model with the new data and visualize the decision boundary...")
    st.write("#### Retraining the model")

    st.code("""
        # extend the dataset and retrain the model
        X_new, y_new = make_classification(n_samples=1000, n_features=2, n_classes=2, n_informative=2, n_redundant=0, n_clusters_per_class=1,class_sep=2.0, flip_y=0, random_state=2001)
    
        df_new = pd.DataFrame(X_new, columns=['feature1', 'feature2'])
        df_new['target'] = y_new
        
        # Combine the old and new datasets
        df_combined = pd.concat([dataset_loader.data, df_new], ignore_index=True)
        df_combined.to_csv("dataset/combined_dataset.csv", index=False)

        # Create an instance of CsvDatasetLoader
        dataset_loader_new = CsvDatasetLoader(csv="dataset/combined_dataset.csv", target_column=target_column)
        X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(dataset_loader_new.data.drop(columns=[target_column]), dataset_loader_new.data[target_column], test_size=0.2, random_state=2001)

        model_new = copy.deepcopy(task.model)
        model_new.train(X_train_new, y_train_new, epochs=100)
        accuracy = model_new.compute_accuracy(X_test_new.values, y_test_new.values)
        """, language="python", line_numbers=True)


    if st.button('Retrain Neural Network'):
        with st.spinner("Retraining the neural network..."):
            time.sleep(2)
            st.write(f"Retraining completed! Model Accuracy: 99.75%")

            st.write(f"Now that the model has been retrained, we can visualize the new decision boundary of the model in the test dataset, and check if the previously computed counterfactual explanations are still valid...")
            # Create columns to center the plot
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(f"images/decision_boundary_with_ces_BinaryLinearSearch_KDTreeNNCE_MCE_NNCE_Wachter_RNCE_after_retraining.png", use_container_width=True, caption="Decision boundary with CEs after retraining")

            st.write("As we can see, the decision boundary of the model has changed after retraining. This change in the decision boundary invalidates almost all the previously computed counterfactual explanations generated through not-robust methods. This highlights the importance of generating robust counterfactual explanations that are stable against model changes.")

            st.write("Importantly, we notice that MCE CE is still valid after retraining. Let's see how to evaluate robust counterfactual explanations using ***RobustX***...")


        

        
def get_evaluation_part():
    st.markdown(
        "<hr style='border: 1px solid black; margin-top: -10px; margin-bottom: 10px;'>", 
        unsafe_allow_html=True)
    st.write("### Step 4: Evaluate CEs against Model Changes")


    _, col2, _ = st.columns([1,2,1])  # Middle column is wider
    with col2:
        st.image("images/eval.png", use_container_width=True, caption="How to evaluate a Counterfactual Exaplanation (CE) with ***RobustX***?")


    st.write("RobustX provides a way to easily evaluate the robustness of counterfactual explanations against model changes. Here, for example, we can evaluate the robustness of a counterfactual explanation by using the Δ-Robustness metric, which measures the maximum change in the model's decision boundary that the counterfactual explanation can withstand while still remaining valid:")

    st.code("""
        from robustx.robustness_evaluations.ApproximateDeltaRobustnessEvaluator import ApproximateDeltaRobustnessEvaluator
        from robustx.robustness_evaluations.DeltaRobustnessEvaluator import DeltaRobustnessEvaluator
            
        # regenerate the task instance
        task = ClassificationTask(model_new, dataset_loader_new)
            
        # define delta_max for the evaluation
        delta_max = 0.05

        # mce_ce is the counterfactual explanation generated using the MCE method
        # we can decide to evaluate the CE using a probabilistic approach based on sampling or a deterministic approach based on MILP
        evaluator = ApproximateDeltaRobustnessEvaluator(task)
        mce_ce = torch.FloatTensor(mce_ce)
        print(f"The counterfactual {mce_ce} is robust against Δ={delta_max} model changes with a confidence 99.9%" if evaluator.evaluate(mce_ce, desired_outcome=1, delta=0.05) else f"The counterfactual {mce_ce} is NOT robust against Δ={delta_max} model changes with a confidence 99.9%")
        >> The counterfactual [[ 1.34792239 -0.976032]] is robust against Δ=0.05 model changes with a confidence 99.9%

        # provable robustness evaluation
        evaluator = DeltaRobustnessEvaluator(task)
        print(f"The counterfactual {mce_ce} is robust against Δ={delta_max} model changes" if evaluator.evaluate(mce_ce, desired_output=1, delta=0.05) else f"The counterfactual {mce_ce} is NOT robust against Δ={delta_max} model changes")
        >> The counterfactual [[ 1.34792239 -0.976032]] is robust against Δ=0.05 model changes
        """, language="python", line_numbers=True)
    
    st.write("As we can notice, for both the robustness evaluation approaches provided in RobustX, the counterfactual explanation generated using the MCE method is robust against Δ=0.05 model changes.")




    







