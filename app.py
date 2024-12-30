from flask import Flask, request, render_template
import sklearn
import pickle
import pandas as pd

model = pickle.load(open('logremodel.pkl', 'rb'))

app = Flask(__name__)


@app.route('/')
def man():
  return render_template('index.html')


@app.route('/back', methods=['GET'])
def back():
  return render_template('index.html')

@app.route('/predict', methods=['POST'])
def get_form_data():
  Name = request.form.get('Name')
  Sex = request.form.get('Sex')
  Age = request.form.get('Age')
  ChestPainType = request.form.get('ChestPainType')
  RestingBP = request.form.get('RestingBP')
  Chol = request.form.get('Chol')
  
  FastingBS = request.form.get('FastingBS')
  RestingECG = request.form.get('RestingECG')
  MaxHR = request.form.get('MaxHR')
  ExerciseAngina = request.form.get('ExerciseAngina')
  Oldpeak = request.form.get('Oldpeak')
  ST_Slope = request.form.get('ST_Slope')

  def standardize(x, mean, standard_deviation):
    y = (float(x) - float(mean)) / float(standard_deviation)
    return float(y)

  age = standardize(Age, 53.510893, 9.432617)
  rbp = standardize(RestingBP, 132.396514, 18.514154)
  chol = standardize(Chol, 198.799564, 109.384145)
  maxhr = standardize(MaxHR, 136.809368, 25.460334)
  oldpeak = standardize(Oldpeak, 0.887364, 1.066570)

  if Sex == "M":
    sm = 1
    sf = 0
  else:
    if Sex == "F":
      sm = 0
      sf = 1

  if ChestPainType == "ASY":
    cpt_asy = 1
    cpt_ata = 0
    cpt_nap = 0
    cpt_ta = 0
  elif ChestPainType == "ATA":
    cpt_asy = 0
    cpt_ata = 1
    cpt_nap = 0
    cpt_ta = 0
  elif ChestPainType == "NAP":
    cpt_asy = 0
    cpt_ata = 0
    cpt_nap = 1
    cpt_ta = 0
  else:
    if ChestPainType == "TA":
      cpt_asy = 0
      cpt_ata = 0
      cpt_nap = 0
      cpt_ta = 1

  if RestingECG == "LVH":
    recg_lvh = 1
    recg_nor = 0
    recg_st = 0
  elif RestingECG == "Normal":
    recg_lvh = 0
    recg_nor = 1
    recg_st = 0
  else:
    if RestingECG == "ST":
      recg_lvh = 0
      recg_nor = 0
      recg_st = 1

  if ExerciseAngina == "N":
    ean = 1
    eay = 0
  else:
    if ExerciseAngina == "Y":
      ean = 0
      eay = 1

  if ST_Slope == "Down":
    ssd = 1
    ssf = 0
    ssu = 0
  elif ST_Slope == "Flat":
    ssd = 0
    ssf = 1
    ssu = 0
  else:
    if ST_Slope == "Up":
      ssd = 0
      ssf = 0
      ssu = 1

  if int(FastingBS) > 120:
    fbs = 1
  else:
    fbs = 0

  input_df = pd.DataFrame({
    'Age': [float(age)],
    'RestingBP': [float(rbp)],
    'Chol': [float(chol)],
    'FastingBS': [fbs],
    'MaxHR': [float(maxhr)],
    'Oldpeak': [float(oldpeak)],
    'Sex_F': [sf],
    'Sex_M': [sm],
    'ChestPainType_ASY': [cpt_asy],
    'ChestPainType_ATA': [cpt_ata],
    'ChestPainType_NAP': [cpt_nap],
    'ChestPainType_TA': [cpt_ta],
    'RestingECG_LVH': [recg_lvh],
    'RestingECG_Normal': [recg_nor],
    'RestingECG_ST': [recg_st],
    'ExerciseAngina_N': [ean],
    'ExerciseAngina_Y': [eay],
    'ST_Slope_Down': [ssd],
    'ST_Slope_Flat': [ssf],
    'ST_Slope_Up': [ssu]
  })

  # The predict_proba method returns an array with two columns. The first column represents the probability of the target variable being 0, and the second column represents the probability of the target variable being 1.

  result = model.predict_proba(input_df)
  no_hd = round(float(result[0][0] * 100), 1)
  yes_hd = round(float(result[0][1] * 100), 1)

  return render_template('output.html', Name=Name, no_hd=no_hd, yes_hd=yes_hd)
