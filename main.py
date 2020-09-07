from __future__ import print_function, unicode_literals
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PyInquirer import style_from_dict, Token, prompt, Separator
from pprint import pprint
import os
import ClassicLogistic as cl
import ClassicSVC as csvc
import ClassicRandomForest as crf
import CNN as cnn

custom_style_2 = style_from_dict({
    Token.Separator: '#6C6C6C',
    Token.QuestionMark: '#FF9D00 bold',
    Token.Selected: '#FF9D00',
    Token.Pointer: '#FF9D00 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#FF9D00 bold',
    Token.Question: '',
})


custom_style_3 = style_from_dict({
    Token.QuestionMark: '#E91E63 bold',
    Token.Selected: '#673AB7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#2196f3 bold',
    Token.Question: '',
})

style = style_from_dict({
    Token.Separator: '#cc5454',
    Token.QuestionMark: '#673ab7 bold',
    Token.Selected: '#cc5454',  # default
    Token.Pointer: '#673ab7 bold',
    Token.Instruction: '',  # default
    Token.Answer: '#f44336 bold',
    Token.Question: '',
})

questions = [
    {

        'type': 'list',
        'name': 'metodo',
        'message': 'Which method do you want to use?',
        'choices':[
            'Baseline',
            'CNN'
        ],
        'validate': lambda answer: 'You must choose at least one.' \
            if len(answer) == 0 else True
    },
    {
        'type': 'input',
        'name': 'pathBaseline',
        'message': 'Insert Path Folder where .csv files are stored:',
        'when': lambda answers: answers['metodo'] == 'Baseline'
    },
    {
            'type': 'input',
            'name': 'mappeddat',
            'message': 'Insert Path Folder where mapped files are stored:',
            'when': lambda answers: answers['metodo'] == 'CNN'
    },
    {
            'type': 'input',
            'name': 'embeddedp',
            'message': 'Insert Path Folder where embed files are stored:',
            'when': lambda answers: answers['metodo'] == 'CNN'
    },
    {
                'type': 'input',
                'name': 'Baselinedata',
                'message': 'Insert Path Folder where .csv files are stored:',
                'when': lambda answers: answers['metodo'] == 'CNN'
    },
    {
      'type': 'list',
      'name': 'BaselineMethod',
      'message': 'Choose the Classifier:',
      'choices': [
          'LogisticRegression',
          'RandomForest',
          'SVC',
        ],
      'when': lambda answers: answers['metodo'] == 'Baseline' and answers['pathBaseline'] != ''
    },
    {
      'type': 'list',
      'name': 'CNNMethod',
      'message': 'Choose the Classifier:',
      'choices': [
          'LogisticRegression',
          'RandomForest',
        ],
      'when': lambda answers: answers['metodo'] == 'CNN' and answers['mappeddat'] != ''
    },
    {
            'type': 'confirm',
            'message': 'Do you want to enable DS3?',
            'name': 'DS3',
            'default': False,
    },
    {
        'type': 'expand',
        'message': 'Choose the dissimilarity distance:',
        'name': 'distance',
        'default': 'c',
        'choices': [
            {
                'key': '1',
                'name': 'Euclidean',
                'value': 'e'
            },
            {
                'key': '2',
                'name': 'Hamming',
                'value': 'h'
            },
            {
                'key': '3',
                'name': 'Chi-Square',
                'value': 'c'
            },
        ],
        'when': lambda answers: answers['DS3'] and answers['pathBaseline'] != ''
    },
    {
            'type': 'confirm',
            'message': 'Verbose?',
            'name': 'verbose',
            'default': False,
    },
    {
            'type': 'confirm',
            'message': 'Do you want to save plots?',
            'name': 'plot',
            'default': False,
    },


]

answers = prompt(questions, style=custom_style_2)
pprint(answers)




if answers['metodo'] == 'Baseline':
    if answers['BaselineMethod'] == 'LogisticRegression':
        if not answers['DS3']:
            answers['distance'] = ''
        cl.start_run(answers['pathBaseline'], answers['DS3'], answers['distance'], answers['plot'], verbose= answers['verbose'])
    if answers['BaselineMethod'] == 'RandomForest':
        if not answers['DS3']:
            answers['distance'] = ''
        crf.start_run(answers['pathBaseline'], answers['DS3'], answers['distance'], answers['plot'], verbose= answers['verbose'])
    if answers['BaselineMethod'] == 'SVC':
        if not answers['DS3']:
            answers['distance'] = ''
        csvc.start_run(answers['pathBaseline'], answers['DS3'], answers['distance'], answers['plot'], verbose= answers['verbose'])

if answers['metodo'] == 'CNN':
    if not answers['DS3']:
        answers['distance'] = ''

    cnn.start_run(answers['mappeddat'], answers['embeddedp'], answers['Baselinedata'], answers['CNNMethod'], answers['DS3'],answers['verbose'], answers['plot'])


