from flask import Flask, request, jsonify, send_from_directory
import Levenshtein
import json as j
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')

app = Flask(__name__)

stowords = set(stopwords.words('english'))
customsword = stowords.union({'got', 'friend', 'need', 'someone'})

with open('dataset1.json', 'r') as f:
    json_data = j.load(f)

def fresponse(input1, data):
    for item in data:
        if input1 in item['tag'] or input1 in item['patterns']:
            return item['responses']
    return ["I'm sorry, No First-Aid Found. Please Recheck your query"]

def stringsimilar(str1, str2):
    distance = Levenshtein.distance(str1, str2)
    maxilen = max(len(str1), len(str2))
    similar = 1 - (distance / maxilen)
    return similar

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('query').lower()

    tokens = word_tokenize(user_input)
    tagged_tokens = pos_tag(tokens)

    wordtag = [word for word, tag in tagged_tokens if tag.startswith('V') or tag.startswith('N') or tag.startswith('J')]
    taglist = [word for word in wordtag if word.lower() not in customsword]
    if not taglist:
        return jsonify({'response': ["I'm sorry, I didn't understand that. Could you please rephrase?"]})
    
    lematize = WordNetLemmatizer()
    lemmawords = [lematize.lemmatize(word) for word in taglist]
    combwords = "".join(taglist)
    tags = [item['tag'] for item in json_data]
    check = [(combwords, tag, stringsimilar(combwords, tag)) for tag in tags]

    if not check:
        return jsonify({'response': ["I'm sorry, No First-Aid Found. Please Recheck your query"]})

    maxsimi = max(check, key=lambda x: x[2])
    bestmatchT = maxsimi[1]
    cbotresp = fresponse(bestmatchT, json_data)
    
    return jsonify({'response': cbotresp})

if __name__ == '__main__':
    app.run(debug=True)
