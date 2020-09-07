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
        'message': 'Che metodo vuoi utilizzare?',
        'choices':[
            'baseline',
            'CNN'
        ],
        'validate': lambda answer: 'You must choose at least one.' \
            if len(answer) == 0 else True
    },
    {
        'type': 'input',
        'name': 'pathbaseline',
        'message': 'Inserisci il Path della cardella dei file CSV: ',
        'when': lambda answers: answers['metodo'] == 'baseline'
    },
    {
            'type': 'input',
            'name': 'mappeddat',
            'message': 'Inserisci il Path della cartella dei mapped',
            'when': lambda answers: answers['metodo'] == 'CNN'
    },
    {
            'type': 'input',
            'name': 'embeddedp',
            'message': 'Inserisci il Path della cartella dei file embed',
            'when': lambda answers: answers['metodo'] == 'CNN'
    },
    {
                'type': 'input',
                'name': 'baselinedata',
                'message': 'Inserisci il Path della cartella dei file CSV',
                'when': lambda answers: answers['metodo'] == 'CNN'
    },
    {
      'type': 'list',
      'name': 'baselineMethod',
      'message': 'Scegli il classificatore',
      'choices': [
          'LogisticRegression',
          'RandomForest',
          'SVC',
        ],
      'when': lambda answers: answers['metodo'] == 'baseline' and answers['pathbaseline'] != ''
    },
    {
      'type': 'list',
      'name': 'CNNMethod',
      'message': 'Scegli il classificatore',
      'choices': [
          'LogisticRegression',
          'RandomForest',
        ],
      'when': lambda answers: answers['metodo'] == 'CNN' and answers['mappeddat'] != ''
    },
    {
            'type': 'confirm',
            'message': 'Vuoi utilizzare la DS3?',
            'name': 'DS3',
            'default': False,
    },
    {
        'type': 'expand',
        'message': 'Scegli la distanza da utilizzare: ',
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
        'when': lambda answers: answers['DS3'] and answers['pathbaseline'] != ''
    },
    {
            'type': 'confirm',
            'message': 'Verbose?',
            'name': 'verbose',
            'default': False,
    },
    {
            'type': 'confirm',
            'message': 'Plot?',
            'name': 'plot',
            'default': False,
    },


]

answers = prompt(questions, style=custom_style_2)
pprint(answers)




if answers['metodo'] == 'baseline':
    if answers['baselineMethod'] == 'LogisticRegression':
        if not answers['DS3']:
            answers['distance'] = ''
        cl.start_run(answers['pathbaseline'], answers['DS3'], answers['distance'], answers['plot'], verbose= answers['verbose'])
    if answers['baselineMethod'] == 'RandomForest':
        if not answers['DS3']:
            answers['distance'] = ''
        crf.start_run(answers['pathbaseline'], answers['DS3'], answers['distance'], answers['plot'], verbose= answers['verbose'])
    if answers['baselineMethod'] == 'SVC':
        if not answers['DS3']:
            answers['distance'] = ''
        csvc.start_run(answers['pathbaseline'], answers['DS3'], answers['distance'], answers['plot'], verbose= answers['verbose'])

if answers['metodo'] == 'CNN':
    if not answers['DS3']:
        answers['distance'] = ''

    cnn.start_run(answers['mappeddat'], answers['embeddedp'], answers['baselinedata'], answers['CNNMethod'], answers['DS3'],answers['verbose'], answers['plot'])


