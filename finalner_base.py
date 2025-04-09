!pip install spacy==2.3.1

import random
import spacy
import json

from google.colab import drive
drive.mount('/content/drive')

import json

train_data=[]

f = open("/content/drive/MyDrive/Resume Parser/data.json", encoding="utf-8")

for i in f:
    ents_list = []
    main_data = []
    ents_dict = dict()
    data = json.loads(i)
    content=data["content"]
    main_data.append(content)
    for annotation in data["annotation"]:
      ents=[]
      ents_dict = dict()
      label = annotation['label']
      points = annotation['points']
      if label==["Designation"] or label==["Companies worked at"] or label==["Skills"]:
        start = points[0]['start']
        end = points[0]['end']
        ents.append(start)
        ents.append(end)
        ents.append(label[0])
        ents_list.append(ents)
        ents_dict['entities'] = ents_list
        main_data.append(ents_dict)
    f_main_data = [main_data[0], main_data[-1]]
    if type(f_main_data[1]) == dict:
      train_data.append(f_main_data)
f.close()

print(train_data)
#with open('training_data.json', 'w', encoding="utf-8") as f:
    #f.write(str(train_data))

def clean_entities(training_data):
    
    clean_data = []
    for text, annotation in training_data:
        
        entities = annotation.get('entities')
        entities_copy = entities.copy()
        
        # append entity only if it is longer than its overlapping entity
        i = 0
        for entity in entities_copy:
            j = 0
            for overlapping_entity in entities_copy:
                # Skip self
                if i != j:
                    e_start, e_end, oe_start, oe_end = entity[0], entity[1], overlapping_entity[0], overlapping_entity[1]
                    # Delete any entity that overlaps, keep if longer
                    if ((e_start >= oe_start and e_start <= oe_end) \
                    or (e_end <= oe_end and e_end >= oe_start)) \
                    and ((e_end - e_start) <= (oe_end - oe_start)):
                        entities.remove(entity)
                j += 1
            i += 1
        clean_data.append((text, {'entities': entities}))
                
    return clean_data

data = clean_entities(train_data)

print(data)

def train_spacy(data, iterations):
    TRAIN_DATA = data
    nlp = spacy.blank('en')
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for itn in range(iterations):
            print('Starting iteration ' + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],
                    [annotations],
                    drop=0.2,
                    sgd=optimizer,
                    losses=losses
                )
                print(losses)
    return nlp

import warnings
warnings.filterwarnings('ignore')
warnings.warn('DelftStack')
warnings.warn('Do not show this message')

nlp = train_spacy(data, 10)

nlp.to_disk('/content/drive/MyDrive/Resume Parser/nlp_ner_model')

nlp_model = spacy.load('/content/drive/MyDrive/Resume Parser/nlp_ner_model')

doc = nlp_model(text)
for ent in doc.ents:
    print(f"{ent.label_.upper():{30}}-{ent.text}")
