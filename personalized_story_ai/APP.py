from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

# Load GPT-2 tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Initialize sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get mood from user input text
def get_mood(text):
    scores = analyzer.polarity_scores(text)
    compound = scores['compound']
    if compound >= 0.5:
        return "happy"
    elif compound <= -0.5:
        return "sad"
    else:
        return "neutral"

# Function to create prompt for story generation
def create_prompt(name, favorite_thing, mood, style):
    prompt = f"Write a {style} story about a person named {name} who loves {favorite_thing}. The story should feel {mood}."
    return prompt

# Function to generate story from prompt using GPT-2
def generate_story(prompt, max_length=150):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        do_sample=True, 
        top_p=0.95, 
        top_k=60,
        no_repeat_ngram_size=2,
        early_stopping=True
    )
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return story

# Flask route for homepage
@app.route('/', methods=['GET', 'POST'])
def home():
    story = ""
    if request.method == 'POST':
        name = request.form['name']
        fav_thing = request.form['fav_thing']
        user_mood_input = request.form['user_mood']
        style = request.form['style']

        mood = get_mood(user_mood_input)
        prompt = create_prompt(name, fav_thing, mood, style)
        story = generate_story(prompt)

    return render_template('index.html', story=story)

if __name__ == "__main__":
    app.run(debug=True)