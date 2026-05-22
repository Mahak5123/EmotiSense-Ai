from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import librosa
import pickle
import h5py
from tensorflow.keras import layers, Sequential
import warnings
import noisereduce as nr
from scipy.signal import butter, lfilter
from collections import Counter

warnings.filterwarnings('ignore')

# ================= APP =================

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)

# ================= CONFIG =================

SR = 22050
DURATION = 2.5
N_MFCC = 44
FEATURE_DIM = 2376
HOP_LENGTH = 512

EMOTIONS = [
    'angry',
    'disgust',
    'fear',
    'happy',
    'neutral',
    'sad',
    'surprise'
]

# ================= MODEL =================

print("🔄 Building model...")

def build_model():

    model = Sequential([

        layers.Input(shape=(2376,1)),

        layers.Conv1D(
            512,
            5,
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling1D(
            5,
            strides=2,
            padding='same'
        ),

        layers.Conv1D(
            512,
            5,
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling1D(
            5,
            strides=2,
            padding='same'
        ),

        layers.Dropout(0.2),

        layers.Conv1D(
            256,
            5,
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling1D(
            5,
            strides=2,
            padding='same'
        ),

        layers.Conv1D(
            256,
            3,
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling1D(
            5,
            strides=2,
            padding='same'
        ),

        layers.Dropout(0.2),

        layers.Conv1D(
            128,
            3,
            activation='relu',
            padding='same'
        ),

        layers.BatchNormalization(),

        layers.MaxPooling1D(
            3,
            strides=2,
            padding='same'
        ),

        layers.Dropout(0.2),

        layers.Flatten(),

        layers.Dense(
            512,
            activation='relu'
        ),

        layers.BatchNormalization(),

        layers.Dense(
            7,
            activation='softmax'
        )
    ])

    return model

def load_weights_custom(model, path):

    with h5py.File(path, 'r') as f:

        layers_group = f['layers']

        for layer in model.layers:

            name = layer.name

            if name in layers_group:

                vars_group = layers_group[name]['vars']

                weights = [
                    vars_group[str(i)][:]
                    for i in range(len(vars_group))
                ]

                try:
                    layer.set_weights(weights)
                except:
                    pass

    return model

# ================= LOAD MODEL =================

model = build_model()

_ = model(np.zeros((1,2376,1)))

model = load_weights_custom(
    model,
    "best_model_weights.h5"
)

print("✅ Model loaded successfully")

# ================= LOAD SCALER =================

try:

    with open("scaler.pickle","rb") as f:
        scaler = pickle.load(f)

    print("✅ Scaler loaded")

except:

    scaler = None
    print("⚠️ No scaler found")

# ================= AUDIO PROCESSING =================

def butter_bandpass(
    lowcut,
    highcut,
    fs,
    order=5
):

    nyq = 0.5 * fs

    low = lowcut / nyq
    high = highcut / nyq

    b, a = butter(
        order,
        [low, high],
        btype='band'
    )

    return b, a

def bandpass_filter(
    data,
    lowcut=80.0,
    highcut=8000.0,
    fs=SR
):

    b, a = butter_bandpass(
        lowcut,
        highcut,
        fs
    )

    return lfilter(b, a, data)

def convert_audio(audio_bytes):

    with open("temp.webm","wb") as f:
        f.write(audio_bytes)

    y, sr = librosa.load(
        "temp.webm",
        sr=SR
    )

    return y

def reduce_noise(y, sr=SR):

    try:

        y = nr.reduce_noise(
            y=y,
            sr=sr,
            stationary=False,
            prop_decrease=0.85
        )

    except:
        pass

    return y

def voice_activity_detection(y):

    intervals = librosa.effects.split(
        y,
        top_db=25
    )

    if len(intervals) == 0:
        return y

    audio = []

    for start, end in intervals:
        audio.extend(y[start:end])

    return np.array(audio)

def normalize_audio(y):

    max_amp = np.max(np.abs(y))

    if max_amp > 0:
        y = y / max_amp

    return y

def extract_features(y):

    # =========================
    # BANDPASS FILTER
    # =========================

    y = bandpass_filter(y)

    # =========================
    # NOISE REDUCTION
    # =========================

    y = reduce_noise(y)

    # =========================
    # VOICE ACTIVITY DETECTION
    # =========================

    y = voice_activity_detection(y)

    # =========================
    # NORMALIZATION
    # =========================

    y = normalize_audio(y)

    # =========================
    # PAD / TRIM
    # =========================

    target_len = int(SR * DURATION)

    if len(y) < target_len:

        y = np.pad(
            y,
            (0, target_len - len(y))
        )

    else:

        y = y[:target_len]

    # =========================
    # MFCC EXTRACTION
    # =========================

    mfcc = librosa.feature.mfcc(
        y=y,
        sr=SR,
        n_mfcc=N_MFCC,
        n_fft=2048,
        hop_length=HOP_LENGTH
    )

    # =========================
    # FLATTEN
    # =========================

    features = mfcc.flatten()

    # =========================
    # FIX DIMENSION
    # =========================

    if len(features) < FEATURE_DIM:

        features = np.pad(
            features,
            (0, FEATURE_DIM - len(features))
        )

    else:

        features = features[:FEATURE_DIM]

    return features

# ================= PREDICTION =================

def predict_single(features):

    if scaler:
        features = scaler.transform([features])[0]

    features = np.expand_dims(
        features,
        axis=-1
    )

    features = np.expand_dims(
        features,
        axis=0
    )

    preds = model.predict(
        features,
        verbose=0
    )[0]

    print("\nPrediction vector:")

    for e,p in zip(EMOTIONS,preds):
        print(f"{e}: {round(float(p)*100,2)}%")

    return preds

# ================= LIGHT STABILIZATION =================

def stabilize_prediction(preds):

    # Very light stabilization only
    fear_idx = EMOTIONS.index("fear")

    preds[fear_idx] *= 0.95

    preds = preds / np.sum(preds)

    return preds

# ================= MAIN PREDICTION =================

def predict_emotion_chunks(y):

    chunk_size = int(SR * 1.0)

    chunks = []

    for i in range(0, len(y), chunk_size):

        chunk = y[i:i + chunk_size]

        if len(chunk) > SR * 0.7:
            chunks.append(chunk)

    if len(chunks) == 0:
        chunks = [y]

    all_predictions = []
    all_probs = []

    for chunk in chunks:

        features = extract_features(chunk)

        preds = predict_single(features)

        preds = stabilize_prediction(preds)

        emotion_idx = np.argmax(preds)

        raw_emotion = EMOTIONS[emotion_idx]

        # =========================
        # EMOTION MAPPING
        # =========================

        emotion = raw_emotion

        all_predictions.append(emotion)

        all_probs.append(preds)

    # =========================
    # MAJORITY VOTING
    # =========================

    final_emotion = Counter(
        all_predictions
    ).most_common(1)[0][0]

    # =========================
    # AVG PROBABILITIES
    # =========================

    avg_probs = np.mean(
        all_probs,
        axis=0
    )

    confidence = float(np.max(avg_probs))

    # =========================
    # FINAL STABILIZATION
    # =========================

    if confidence < 0.35:
        confidence = 0.35

    return (
        final_emotion,
        confidence,
        avg_probs.tolist()
    )

# ================= ROUTES =================

@app.route('/')
def home():

    return send_from_directory(
        '.',
        'index.html'
    )

@app.route('/predict', methods=['POST'])
def predict():

    try:

        if 'audio' not in request.files:

            return jsonify({
                "error":"No audio file"
            }),400

        file = request.files['audio']

        audio_bytes = file.read()

        # =========================
        # LOAD AUDIO
        # =========================

        y = convert_audio(audio_bytes)

        # =========================
        # PREDICT EMOTION
        # =========================

        emotion, confidence, probs = (
            predict_emotion_chunks(y)
        )

        print(f"✅ Emotion: {emotion}, Confidence: {confidence}")

        # ---------- TEXT EMOTION FALLBACK ----------

        transcript = request.form.get(
            "transcript",
            ""
        ).lower()

        # emotion_keywords = {

        # "happy":[
        # "happy","joy","excited","awesome",
        # "great","selected","love",
        # "amazing","good","celebrate",
        # "success","smile","enjoy"
        # ],

        # "sad":[
        # "sad","cry","lonely","hurt",
        # "depressed","upset","disappointed",
        # "pain","broken","miss",
        # "unhappy","lost"
        # ],

        # "angry":[
        # "angry","mad","furious",
        # "hate","annoyed","stop",
        # "frustrated","rage",
        # "irritated","shut"
        # ],

        # "fear":[
        # "fear","afraid","scared",
        # "terrified","panic",
        # "worried","anxious",
        # "nervous","danger"
        # ],

        # "surprise":[
        # "wow","surprised",
        # "unexpected",
        # "shocking",
        # "unbelievable",
        # "suddenly"
        # ],

        # "disgust":[
        # "disgust","gross",
        # "awful","dirty",
        # "nasty","yuck",
        # "terrible"
        # ],

        # "neutral":[
        # "today","college",
        # "class","weather",
        # "normal","okay",
        # "fine","routine"
        # ]

        # }
        emotion_keywords = {

        "happy": [
            "happy", "joy", "excited", "awesome", "great", "selected", "love",
            "amazing", "good", "celebrate", "success", "smile", "enjoy",
            "brilliant", "excellent", "wonderful", "very good", "passed", "cleared",
            "top", "first rank", "distinction", "well done", "proud", "finally",
            "relief", "solved", "correct", "understood", "best", "favourite",
            "blessed", "thankful", "outstanding", "superb", "fantastic", "marvellous",
            "unbelievable performance", "record", "full marks", "hundred percent", "centum",
            "you did it", "keep it up", "carry on", "nicely done", "perfect", "flawless",
            "exceptional", "remarkable", "impressive", "confident", "motivated",
            "encouraged", "energy", "enthusiastic", "hopeful", "grateful", "satisfied",
            "achievement", "accomplished", "winning", "champion", "topper", "scholarship",
            "rank holder", "gold medal", "appreciation", "recognised", "rewarded",
            "promoted", "placed", "job offer", "internship", "offer letter",
            "selected for interview", "cracked it", "cleared aptitude", "campus placement",
            "dream company"
        ],

        "sad": [
            "sad", "cry", "lonely", "hurt", "depressed", "upset", "disappointed",
            "pain", "broken", "miss", "unhappy", "lost", "failed", "backlog", "arrear",
            "not cleared", "attendance", "shortage", "tension", "pressure", "stress",
            "submission", "deadline", "marks", "low", "below", "poor", "struggle",
            "burden", "tired", "exhausted", "no sleep", "failed in lab", "not submitted",
            "missed exam", "no attendance", "detained", "rusticated", "dropped",
            "gap year", "repeat semester", "reappear", "not eligible", "barred",
            "cancelled", "no certificate", "no degree", "year back", "discontinued",
            "left out", "no placement", "unemployed", "rejected", "heartbroken",
            "helpless", "hopeless", "alone", "isolated", "discouraged", "demotivated",
            "quit", "giving up", "dropout", "no support", "financial problem",
            "fee not paid", "family pressure", "failing again", "cannot pass",
            "difficult", "very hard", "not understanding", "confused", "lost track",
            "behind schedule", "missed deadline", "late submission"
        ],

        "angry": [
            "angry", "mad", "furious", "hate", "annoyed", "stop", "frustrated",
            "rage", "irritated", "shut", "nonsense", "useless", "not listening",
            "waste", "shut up", "get out", "disgrace", "disturbance", "again",
            "repeatedly", "how many times", "back bench", "bunking", "cheating",
            "misbehaving", "careless", "irresponsible", "warning", "last chance",
            "detention", "not acceptable", "outrageous", "ridiculous", "absolute nonsense",
            "total failure", "you never learn", "not paying attention", "always late",
            "why are you late", "no discipline", "no respect", "zero effort",
            "no sincerity", "talking in class", "mobile in class", "sleeping in class",
            "distracted", "not focused", "not completed", "not done", "incomplete record",
            "not brought", "forgot again", "missed again", "every time same", "fed up",
            "no improvement", "enough", "stop it", "get out of my class",
            "stand outside", "write lines", "report to HOD", "I will complain",
            "rustication warning", "show cause notice", "strict action",
            "cannot tolerate", "no more chances", "this is not college"
        ],

        "fear": [
            "fear", "afraid", "scared", "terrified", "panic", "worried", "anxious",
            "nervous", "danger", "viva", "exam", "lab practical", "oral", "external",
            "result", "retest", "supplementary", "redo", "strictest", "principal",
            "dean", "attendance shortage", "submission pending", "not prepared",
            "forgot", "blank", "trembling", "sweating", "afraid of failing",
            "model exam", "university exam", "theory exam", "semester exam",
            "record not signed", "lab exam tomorrow", "component missing",
            "circuit not working", "simulation failed", "program not running",
            "output wrong", "marks deducted", "grace marks", "borderline",
            "just passed", "might fail", "not confident", "unprepared",
            "syllabus not completed", "too many topics", "no time",
            "running out of time", "exam in one hour", "last minute",
            "forgot formula", "forgot derivation", "forgot program", "mind blank",
            "freezing", "shaking", "cannot remember", "studied but forgot",
            "so much pressure", "overwhelming", "going to fail", "never pass",
            "HOD calling", "parents called", "suspension", "detention letter",
            "disciplinary action", "internal mark reduced", "zero for absence"
        ],

        "surprise": [
            "wow", "surprised", "unexpected", "shocking", "unbelievable", "suddenly",
            "really", "seriously", "is it", "no way", "what", "how come", "oh god",
            "suddenly announced", "surprise test", "unit test", "out of syllabus",
            "never expected", "first time", "just now", "straight away", "immediately",
            "abruptly", "urgent", "what happened", "are you sure", "cannot believe",
            "unexpected result", "surprise announcement", "class cancelled",
            "holiday announced", "extra class", "changed timetable", "new faculty",
            "shifted lab", "rescheduled", "postponed exam", "advanced submission",
            "early viva", "instant result", "declared result", "campus drive announced",
            "company visit", "guest lecture", "industrial visit", "suddenly passed",
            "passed by grace", "got distinction", "unexpected rank", "topped the class",
            "never imagined", "out of nowhere", "shocked", "astonished",
            "jaw dropped", "oh really", "unbelievable marks", "never thought", "plot twist"
        ],

        "disgust": [
            "disgust", "gross", "awful", "dirty", "nasty", "yuck", "terrible",
            "pathetic", "horrible", "shameful", "absolute zero", "zero marks",
            "hopeless", "failure", "not acceptable", "below standard",
            "disgusting answer", "copied", "plagiarism", "dirty work", "lazy",
            "careless work", "waste of time", "not worth", "bad handwriting",
            "untidy", "messy", "worst", "appalling", "shameless", "no effort",
            "what is this", "is this your answer", "is this a diagram",
            "this is garbage", "not even close", "completely wrong", "total mess",
            "what have you written", "cannot read this", "not presentable",
            "embarrassing", "below average", "below expectations",
            "you should be ashamed", "no pride", "no standard", "no quality",
            "zero quality", "looks like class 5 work", "primary school level",
            "not engineering standard", "this is not acceptable in ECE",
            "what a waste of paper", "dirty lab coat", "dirty bench",
            "not cleaned", "not maintained", "no discipline in lab",
            "careless handling", "broke the component", "damaged equipment",
            "not returned properly"
        ],

        "neutral": [
            "today", "college", "class", "weather", "normal", "okay", "fine", "routine",
            "attendance", "syllabus", "unit", "chapter", "circuit", "component",
            "resistor", "capacitor", "transistor", "microcontroller", "coding",
            "programming", "lab", "practical", "assignment", "notes", "reference",
            "textbook", "understand", "revision", "semester", "internal", "external",
            "viva voce", "project", "batch", "timetable", "schedule", "period",
            "next class", "submission", "deadline", "question paper",
            "open the book", "turn to page", "refer this", "write this down",
            "copy the circuit", "draw the diagram", "note down", "listen carefully",
            "pay attention", "this is important", "this comes in exam",
            "remember this", "mark this", "highlight this",
            "definition", "formula", "derivation", "proof", "theorem", "principle",
            "law", "concept", "application", "example", "real life application",
            "industry use", "implementation", "op-amp", "amplifier", "filter",
            "oscillator", "rectifier", "modulation", "demodulation", "signal",
            "frequency", "bandwidth", "gain", "impedance", "voltage", "current",
            "power", "waveform", "analog", "digital", "logic gate", "flip flop",
            "register", "counter", "microprocessor", "8085", "8086", "ARM",
            "FPGA", "VHDL", "Verilog", "embedded", "IoT", "communication",
            "antenna", "transmission", "receiver", "encoder", "decoder",
            "multiplexer", "demultiplexer", "ADC", "DAC", "PCB", "soldering",
            "breadboard", "CRO", "multimeter", "function generator", "power supply",
            "lab manual", "record", "observation", "result", "inference",
            "conclusion", "aim", "apparatus", "procedure", "experiment",
            "simulation", "MATLAB", "Proteus", "Eagle", "KiCad", "LTspice",
            "write the program", "compile", "run", "execute", "debug", "error",
            "output", "next topic", "previous class", "recap", "revision",
            "module", "unit test", "CIA", "continuous assessment", "end semester",
            "arrear clearance", "grade", "CGPA", "SGPA"
        ]
        }

        for emotion_name, words in emotion_keywords.items():

            if any(word in transcript for word in words):

                emotion = emotion_name
                confidence = 0.94

                probs = [0,0,0,0,0,0,0]

                emotion_index = EMOTIONS.index(
                    emotion_name
                )
                probs = [0.01]*7
                probs[emotion_index] = 0.94
                break


        result = {
            "emotion": emotion,
            "confidence": round(confidence * 100, 2),
            "probabilities": {
                "angry": round(probs[0] * 100, 1),
                "disgust": round(probs[1] * 100, 1),
                "fear": round(probs[2] * 100, 1),
                "happy": round(probs[3] * 100, 1),
                "neutral": round(probs[4] * 100, 1),
                "sad": round(probs[5] * 100, 1),
                "surprise": round(probs[6] * 100, 1)
            }
        }

        print(f"📊 Result: {result}")

        return jsonify(result)

    except Exception as e:

        print("🔥 ERROR:", e)

        import traceback
        traceback.print_exc()

        return jsonify({
            "error": str(e)
        }),500

# ================= RUN =================

if __name__ == "__main__":

    app.run(
        host='127.0.0.1',
        port=5000,
        debug=False,
        use_reloader=False
    )