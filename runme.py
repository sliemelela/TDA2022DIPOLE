import processing
import visualizations
import analysis

from pyfiglet import Figlet
from PyInquirer import style_from_dict, Token, prompt, Separator
from prettytable import PrettyTable

if __name__ == "__main__":

    # Intro Banner
    f = Figlet(font='slant')
    print(f.renderText('DIPOLE'))

    # Style of the questions
    style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
    })


    goal_of_program = [
        {
            'type': 'list',
            'message': 'Would you like to visualize or run quantative tests?',
            'name': 'goal',
            'choices': [
                {
                    'name': 'Only visualize'
                },
                {
                    'name': 'Only run quantative tests'
                },
            ],
        }
    ]

    data_type = [
        {
            'type': 'list',
            'message': 'What data set would you like as input?',
            'name': 'data',
            'choices': [
                {
                    'name': 'All data sets from original paper'
                },
                {
                    'name': 'All new data sets'
                },
                {
                    'name': 'Mammoth data set (from original paper)'
                },
                {
                    'name': 'Brain Artery Tree data set (from original paper)'
                },
                {
                    'name': 'Stanford Faces data set (from original paper)'
                },
                {
                    'name': 'Swiss Roll with Holes data set (from original paper)'
                },
                {
                    'name': 'Collection of circles ambient in 3D'
                },
                {
                    'name': 'First hand data set'
                },
                {
                    'name': 'Second hand data set'
                },
                {
                    'name': 'Tracing Heart data set'
                },
                {
                    'name': 'Tracing Heart data set (with outline)'
                },
            ],
        }
    ]

    x = PrettyTable()

    # Retrieving the desired district
    goal_answer = prompt(goal_of_program, style=style)
    
    # Prompting users with which algorithm they want to pick
    data_answer = prompt(data_type, style=style)

    if goal_answer['goal'] == 'Only visualize':
        if data_answer['data'] == 'All data sets from original paper':
            visualizations.mammoth_visualization()
            visualizations.brain_visualization()
            visualizations.swissroll_visualization()
            visualizations.stanford_visualization()

        elif data_answer['data'] == 'All new data sets':
            visualizations.pic_data_visualization("hand-images-1")
            visualizations.pic_data_visualization("hand-images-2")
            visualizations.pic_data_visualization("heart")
            visualizations.pic_data_visualization("heart-guide")
            visualizations.circle_visualization()
        
        elif data_answer['data'] == 'Mammoth data set (from original paper)':
            visualizations.mammoth_visualization()
        elif data_answer['data'] == 'Brain Artery Tree data set (from original paper)':
            visualizations.brain_visualization()
        elif data_answer['data'] == 'Stanford Faces data set (from original paper)':
            visualizations.stanford_visualization()
        elif data_answer['data'] == 'Swiss Roll with Holes data set (from original paper)':
            visualizations.swissroll_visualization()
        elif data_answer['data'] == 'Collection of circles ambient in 3D':
            visualizations.circle_visualization()
        elif data_answer['data'] == 'First hand data set':
            visualizations.pic_data_visualization("hand-images-1")  
        elif data_answer['data'] == 'Second hand data set':
            visualizations.pic_data_visualization("hand-images-2") 
        elif data_answer['data'] == 'Tracing Heart data set':
            visualizations.pic_data_visualization("heart") 
        elif data_answer['data'] == 'Tracing Heart data set (with outline)':
            visualizations.pic_data_visualization("heart-guide") 

    if goal_answer['goal'] == 'Only run quantative tests':
        if data_answer['data'] == 'All data sets from original paper':
            analysis.mammoth_analyse()
            analysis.brain_analyse()
            analysis.swissroll_analyse()
            analysis.stanford_analyse()

        elif data_answer['data'] == 'All new data sets':
            analysis.pic_data_analyse("hand-images-1")
            analysis.pic_data_analyse("hand-images-2")
            analysis.pic_data_analyse("heart")
            analysis.pic_data_analyse("heart-guide")
            analysis.circle_analyse()
        
        elif data_answer['data'] == 'Mammoth data set (from original paper)':
            analysis.mammoth_analyse()
        elif data_answer['data'] == 'Brain Artery Tree data set (from original paper)':
            analysis.brain_analyse()
        elif data_answer['data'] == 'Stanford Faces data set (from original paper)':
            analysis.stanford_analyse()
        elif data_answer['data'] == 'Swiss Roll with Holes data set (from original paper)':
            analysis.swissroll_analyse()
        elif data_answer['data'] == 'Collection of circles ambient in 3D':
            analysis.circle_analyse()
        elif data_answer['data'] == 'First hand data set':
            analysis.pic_data_analyse("hand-images-1")  
        elif data_answer['data'] == 'Second hand data set':
            analysis.pic_data_analyse("hand-images-2") 
        elif data_answer['data'] == 'Tracing Heart data set':
            analysis.pic_data_analyse("heart") 
        elif data_answer['data'] == 'Tracing Heart data set (with outline)':
            analysis.pic_data_analyse("heart-guide") 
        

    print("Check the figures folder for the output!")

    