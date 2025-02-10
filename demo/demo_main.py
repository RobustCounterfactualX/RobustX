import streamlit as st
from utils_demo import *



# Streamlit App
def main():

    ########## INTRO #########
    get_intro()
   
    
   ########## TASK #########
    get_task_part()
   

    ########## GENERATION #########
    get_generation_part()

    ########## RETRAINING #########
    get_retraining_part()

    ########## EVALUATION #########
    get_evaluation_part()


if __name__ == "__main__":
    main()
