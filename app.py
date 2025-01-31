import json
import os
import pickle
from flask import Flask, request,Response
from flask_api import status
import matplotlib
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # basic plotting
matplotlib.use('Agg')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random
from matplotlib.colors import ListedColormap

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from mlxtend.plotting import plot_confusion_matrix
import seaborn as sns # more plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, precision_score, \
recall_score, cohen_kappa_score, classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from flask_cors import CORS


##use amazon database for storage of models & csv files
import boto3
from botocore.exceptions import NoCredentialsError
from io import BytesIO

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get("SECRET_KEY", None)
CORS(app)


def download_data_from_s3(bucket_name, s3_file_key):
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    obj = s3.get_object(Bucket=bucket_name, Key=s3_file_key)
    return pd.read_csv(BytesIO(obj['Body'].read()))

def upload_pickle_to_s3(data, bucket_name, s3_file_key):
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    with BytesIO() as f:
        data.to_pickle(f)
        f.seek(0)
        s3.upload_fileobj(f, bucket_name, s3_file_key)


def download_pickle_from_s3(bucket_name, s3_file_key):
    s3 = boto3.client('s3',
                      aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
                      aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))
    obj = s3.get_object(Bucket=bucket_name, Key=s3_file_key)
    return pd.read_pickle(BytesIO(obj['Body'].read()))

def data_cleaning_feature1():
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    events_file_key = 'Files/events.csv'
    ginf_file_key = 'Files/ginf.csv'

    # Download data from S3
    events = download_data_from_s3(bucket_name, events_file_key)
    ginf = download_data_from_s3(bucket_name, ginf_file_key)

    # Define mappings for cleaning
    event_types = {1:'Attempt', 2:'Corner', 3:'Foul', 4:'Yellow card', 5:'Second yellow card', 6:'Red card', 7:'Substitution', 8:'Free kick won', 9:'Offside', 10:'Hand ball', 11:'Penalty conceded'}
    event_types2 = {12:'Key Pass', 13:'Failed through ball', 14:'Sending off', 15:'Own goal'}
    sides = {1:'Home', 2:'Away'}
    shot_places = {1:'Bit too high', 2:'Blocked', 3:'Bottom left corner', 4:'Bottom right corner', 5:'Centre of the goal', 6:'High and wide', 7:'Hits the bar', 8:'Misses to the left', 9:'Misses to the right', 10:'Too high', 11:'Top centre of the goal', 12:'Top left corner', 13:'Top right corner'}
    shot_outcomes = {1:'On target', 2:'Off target', 3:'Blocked', 4:'Hit the bar'}
    locations = {1:'Attacking half', 2:'Defensive half', 3:'Centre of the box', 4:'Left wing', 5:'Right wing', 6:'Difficult angle and long range', 7:'Difficult angle on the left', 8:'Difficult angle on the right', 9:'Left side of the box', 10:'Left side of the six yard box', 11:'Right side of the box', 12:'Right side of the six yard box', 13:'Very close range', 14:'Penalty spot', 15:'Outside the box', 16:'Long range', 17:'More than 35 yards', 18:'More than 40 yards', 19:'Not recorded'}
    bodyparts = {1:'right foot', 2:'left foot', 3:'head'}
    assist_methods = {0:pd.NA, 1:'Pass', 2:'Cross', 3:'Headed pass', 4:'Through ball'}
    situations = {1:'Open play', 2:'Set piece', 3:'Corner', 4:'Free kick'}

    # Perform data cleaning and mapping
    events['event_type'] = events['event_type'].map(event_types)
    events['event_type2'] = events['event_type2'].map(event_types2)
    events['side'] = events['side'].map(sides)
    events['shot_place'] = events['shot_place'].map(shot_places)
    events['shot_outcome'] = events['shot_outcome'].map(shot_outcomes)
    events['location'] = events['location'].map(locations)
    events['bodypart'] = events['bodypart'].map(bodyparts)
    events['assist_method'] = events['assist_method'].map(assist_methods)
    events['situation'] = events['situation'].map(situations)

    # Define categorical columns
    cats = ['id_odsp', 'event_type', 'player', 'player2', 'event_team', 'opponent', 'shot_place', 'shot_outcome', 'location', 'bodypart', 'assist_method', 'situation']
    d = {cat: 'category' for cat in cats}

    # Convert columns to categorical
    events = events.astype(d)
    events['is_goal'] = events['is_goal'].astype('bool')

    # Upload processed DataFrame to S3 as a .pkl file
    upload_pickle_to_s3(events, bucket_name, 'Files/models/events_df.pkl')

events=None


def load_models():
    global events
    try:
        events = download_pickle_from_s3(os.environ.get('S3_BUCKET_NAME'), 'Files/models/events_df.pkl')
    except Exception as e:
        print(f"Error loading model from S3: {e}")
        events = None
    return "Loaded"

@app.route('/')
def index():
    return "Hello SCIQ"


@app.route('/feature1', methods=['POST'])
def feature1_output():
    load_models()  # Ensure events DataFrame is loaded from S3
    global events
    
    data = request.get_json()
    team = data['team']

    if events is None:
        return Response(status=500, mimetype='application/json', response=json.dumps({'error': 'Events data not loaded'}))

    filtered_team = events[events['event_team'] == team]

    goals = filtered_team[filtered_team['is_goal'] == True]['time'].value_counts().sort_index()
    substitutions = filtered_team[filtered_team['event_type'] == 'Substitution']['time'].value_counts().sort_index()
    red_cards = filtered_team[filtered_team['event_type'] == 'Red card']['time'].value_counts().sort_index()
    yellow_cards = filtered_team[filtered_team['event_type'].isin(['Yellow card', 'Second yellow card'])]['time'].value_counts().sort_index()

    graph1res = {
        'labels': goals.index.tolist(),
        'data': goals.values.tolist()
    }

    graph2res = {
        'labels': substitutions.index.tolist(),
        'data': substitutions.values.tolist()
    }

    graph3res = {
        'labels': red_cards.index.tolist(),
        'data': red_cards.values.tolist()
    }

    graph4res = {
        'labels': yellow_cards.index.tolist(),
        'data': yellow_cards.values.tolist()
    }

    response = Response(status=200, mimetype='application/json')
    response.data = json.dumps({'graph1': graph1res, 'graph2': graph2res, 'graph3': graph3res, 'graph4': graph4res})
    
    return response



    


events_f2 = None
info_f2 = None
shots = None
model = None
# Function for data cleaning feature 2
def data_cleansing_feature2(bucket_name):
    global events_f2, info_f2, shots

    pd.options.display.max_columns = 999
    pd.options.display.max_rows = 50

    ## bucket_name = os.environ.get('S3_BUCKET_NAME')

    events_file_key = 'Files/events.csv'
    info_file_key = 'Files/ginf.csv'

    # Download CSV files from S3
    events_f2 = download_data_from_s3(bucket_name, events_file_key)
    info_f2 = download_data_from_s3(bucket_name, info_file_key)

    # Merge events and info data
    events_f2 = events_f2.merge(info_f2[['id_odsp', 'country', 'date']], on='id_odsp', how='left')

    # Extract year from date column
    extract_year = lambda x: datetime.strptime(x, "%Y-%m-%d").year
    events_f2['year'] = events_f2['date'].apply(extract_year)

    # Filter shots
    shots = events_f2[events_f2['event_type'] == 1]
    shots['player'] = shots['player'].str.title()
    shots['player2'] = shots['player2'].str.title()
    shots['country'] = shots['country'].str.title()

    return True


# Function to train model and save to S3
def train_model_f2(bucket_name):
    global shots

    data = pd.get_dummies(shots.iloc[:,-8:-3], columns=['location', 'bodypart','assist_method', 'situation'])
    data.columns = ['fast_break', 'loc_centre_box', 'loc_diff_angle_lr', 'diff_angle_left', 'diff_angle_right',
                    'left_side_box', 'left_side_6ybox', 'right_side_box', 'right_side_6ybox', 'close_range',
                    'penalty', 'outside_box', 'long_range', 'more_35y', 'more_40y', 'not_recorded', 'right_foot', 
                    'left_foot', 'header', 'no_assist', 'assist_pass', 'assist_cross', 'assist_header',
                    'assist_through_ball', 'open_play', 'set_piece', 'corner', 'free_kick']
    data['is_goal'] = shots['is_goal']

    X = data.iloc[:,:-1]
    y = data.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.35, random_state=1)

    def evaluate_model(params): 
        model = GradientBoostingClassifier(
                            learning_rate=params['learning_rate'],
                            min_samples_leaf=params['min_samples_leaf'],
                            max_depth = params['max_depth'],
                            max_features = params['max_features']
                            )

        model.fit(X_train, y_train)
        return {
            'learning_rate': params['learning_rate'],
            'min_samples_leaf': params['min_samples_leaf'],
            'max_depth': params['max_depth'],
            'max_features': params['max_features'],
            'train_ROCAUC': roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]),
            'test_ROCAUC': roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]),
            'recall': recall_score(y_test, model.predict(X_test)),
            'precision': precision_score(y_test, model.predict(X_test)),
            'f1_score': f1_score(y_test, model.predict(X_test)),
            'train_accuracy': model.score(X_train, y_train),
            'test_accuracy': model.score(X_test, y_test),
        }

    def objective(params):
        res = evaluate_model(params)
        
        res['loss'] = - res['test_ROCAUC'] # Esta loss es la que hyperopt intenta minimizar
        res['status'] = STATUS_OK # Asi le decimos a hyperopt que el experimento salio bien
        return res 

    hyperparameter_space = {
            'learning_rate': hp.uniform('learning_rate', 0.05, 0.3),
            'min_samples_leaf': hp.choice('min_samples_leaf', range(15, 200)),
            'max_depth': hp.choice('max_depth', range(2, 20)),
            'max_features': hp.choice('max_features', range(3, 27))
    }

    trials = Trials()
    best_params = fmin(
        objective,
        space=hyperparameter_space,
        algo=tpe.suggest,
        max_evals=50,
        trials=trials
    )

    model = GradientBoostingClassifier(
                        learning_rate=best_params['learning_rate'],
                        min_samples_leaf=best_params['min_samples_leaf'],
                        max_depth=best_params['max_depth'],
                        max_features=best_params['max_features']
                        )
    model.fit(X_train, y_train)

    # Save trained model to S3
    model_key = 'Files/models/model.sav'
    upload_pickle_to_s3(bucket_name, model_key, model)

    # Save shots DataFrame with predictions to S3
    shots['prediction'] = model.predict_proba(X)[:, 1]
    shots['difference'] = shots['prediction'] - shots['is_goal']
    shots_key = 'Files/models/shots.pkl'
    upload_pickle_to_s3(bucket_name, shots_key, shots)

    return True

@app.route('/clean_train_model_f2')
def clean_train_f2():
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    data_cleansing_feature2(bucket_name)
    train_model_f2(bucket_name)
    return "Training complete"





@app.route('/feature2', methods=['POST'])
def expected_goal():
    response = Response(mimetype='application/json')

    shots = download_pickle_from_s3(os.environ.get('S3_BUCKET_NAME'), 'Files/models/shots.pkl')

    team_name = "All"
    sub_feature = "Best Finisher"

    if 'team_name' in request.args:
        team_name = request.args.get('team_name')
    if 'sub_feature' in request.args:
        sub_feature = request.args.get('sub_feature')

    print(team_name)
    print("FEATURE NAME: ", sub_feature)

    if team_name == "All":
        shots_team = shots
    else:
        shots_team = shots[shots['event_team'] == team_name]

    players = shots_team.groupby('player').sum().reset_index()
    players.rename(columns={'is_goal': 'trueGoals', 'prediction': 'expectedGoals'}, inplace=True)
    players['expectedGoals'] = round(players['expectedGoals'], 2)
    players['difference'] = round(players['difference'], 2)
    players['ratio'] = players['trueGoals'] / players['expectedGoals']

    res = []
    if sub_feature == "Best Finisher":
        show = players.sort_values(['difference', 'trueGoals']).reset_index(drop=True)
        show['rank'] = show.index + 1
        show = show[['rank', 'player', 'difference', 'trueGoals', 'expectedGoals']].head(11)

        temp = show.to_dict()
        td = {}
        for i in range(11):
            td['rank'] = temp['rank'][i]
            td['player'] = temp['player'][i]
            td['difference'] = temp['difference'][i]
            td['trueGoals'] = temp['trueGoals'][i]
            td['expectedGoals'] = temp['expectedGoals'][i]
            res.append(td)
            td = {}

        sns.set_style("dark")
        fig, ax = plt.subplots(figsize=[12, 5])
        ax = sns.barplot(x=abs(show['difference']), y=show['player'], palette='viridis', alpha=0.9)
        ax.set_xticks(np.arange(0, 65, 5))
        ax.set_xlabel(xlabel='Diff. between Goals Scored and Goals Expected', fontsize=12)
        ax.set_ylabel(ylabel='')
        ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=12)
        plt.title("Best Finishers: most goals on top of expected", fontsize=20, fontfamily='serif')
        ax.grid(color='black', linestyle='-', linewidth=0.1, alpha=0.8, axis='x')
        ##plt.savefig('/Files/images/F2_BestFinishers.png')

        response.status = status.HTTP_200_OK
        response.data = json.dumps({'result': res})
        return response

    elif sub_feature == "Most Expected Goals":
        show = players[['player', 'trueGoals', 'expectedGoals']].sort_values(['expectedGoals'], ascending=False).head(10)
        temp = show.to_dict(orient='records')  # Convert to list of dictionaries
        res = []

        for i, entry in enumerate(temp):
            td = {
                'player': entry['player'],
                'trueGoals': entry['trueGoals'],
                'expectedGoals': entry['expectedGoals']
            }
            res.append(td)

        response.status = status.HTTP_200_OK
        response.data = json.dumps({"result": res})
        return response

    elif sub_feature == "Outside The Box":
        outside_box = shots[shots['location'] == 15]
        outbox_players = outside_box.groupby('player').sum().reset_index()
        outbox_players.rename(columns={'event_type': 'n_outbox_shots', 'is_goal': 'trueGoals', 'prediction': 'expectedGoals'}, inplace=True)
        show = outbox_players.sort_values(['difference', 'trueGoals']).reset_index(drop=True)
        show['rank'] = show.index + 1
        show = show[['rank', 'player', 'n_outbox_shots', 'trueGoals', 'expectedGoals', 'difference']].head(10)
        temp = show.to_dict()
        td = {}
        for i in range(len(temp['player'])):
            td['rank'] = temp['rank'][i]
            td['player'] = temp['player'][i]
            td['n_outbox_shots'] = temp['n_outbox_shots'][i]
            td['difference'] = temp['difference'][i]
            td['trueGoals'] = temp['trueGoals'][i]
            td['expectedGoals'] = temp['expectedGoals'][i]
            res.append(td)
            td = {}

        response.status = status.HTTP_200_OK
        response.data = json.dumps({'result': res})
        return response

    else:
        pass











def get_best_team(nationality, chosen_tactic, cc):
    print('=======================================')
    print(nationality, chosen_tactic, cc)
    
    # Load players data from S3
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    players_file_key = "Files/players_22.csv"
    df = download_data_from_s3(bucket_name, players_file_key)
    
    df.dob = pd.to_datetime(df.dob)
    df.loc[:, 'main_position'] = df['player_positions'].apply(lambda x: x.split(',')[0])
    data = df
    best_team = []
    shortlisted = []
    
    for i in chosen_tactic['positions']:
        if cc == "country":
            print("call on country")
            potential_players = data[(data['nationality_name'] == nationality) & (data['player_positions'].str.contains(i))].sort_values(['overall'], ascending=False)
        else:
            print("call on club")
            potential_players = data[(data['club_name'] == nationality) & (data['player_positions'].str.contains(i))].sort_values(['overall'], ascending=False)
        
        print(potential_players.head())
        
        try:
            ind = 0
            while potential_players.iloc[ind].short_name in shortlisted:
                ind += 1
            
            shortlisted.append(potential_players.iloc[ind].short_name)
            best_team.append({
                'position': i,
                'Name': potential_players.iloc[ind].short_name,
                'club_name': potential_players.iloc[ind].club_name,
                'Faceurl': potential_players.iloc[ind].player_face_url,
                'Overall': int(potential_players.iloc[ind].overall),
                'Potential': int(potential_players.iloc[ind].potential),
                'Age': int(potential_players.iloc[ind].age),
                'height_cm': float(potential_players.iloc[ind].height_cm),
                'weight_kg': float(potential_players.iloc[ind].weight_kg)
            })

        except Exception as e:
            best_team.append({
                'position': i,
                'Name': "",
                'club_name': "",
                'Faceurl': "",
                'Overall': "",
                'Potential': "",
                'Age': "",
                'height_cm': "",
                'weight_kg': ""
            })
    
    print(best_team)
    return best_team





@app.route('/feature3',methods = ['POST'])
def feature3():
    response = Response(mimetype='application/json')



    tactic_433 = {
        'name': "433",
        'positions': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'RM', 'LW', 'ST', 'RW']
        }

    tactic_442 = {
        'name': "442",
        'positions': ['GK', 'LB', 'CB', 'CB', 'RB', 'LM', 'CDM', 'CDM', 'RM', 'ST', 'ST']
        }

    tactic_352 = {
        'name': "352",
        'positions': ['GK', 'LWB', 'CB', 'RWB', 'LM', 'CDM', 'CAM', 'CM', 'RM', 'LW', 'RW']
        }

    tactic = str(request.args.get('tactic'))

    if tactic == "433":
        send_tactic = tactic_433
    elif tactic == "442":
        send_tactic = tactic_442
    elif tactic == "352":
        send_tactic = tactic_352
    else:
        positionlist = []
        for k in range(11):
            ar = 'position[' + str(k) + '][value]'
            positionlist.append(request.args.get(ar))
        
        print('poslist', positionlist)
        custom_tactic = {
            'name': str(tactic),
            'positions': positionlist
        }
        send_tactic = custom_tactic
    
    print("Send tactic is :", send_tactic)
    
    if 'country' in request.args:
        country = request.args.get('country')
        res = get_best_team(country, send_tactic, 'country')
    else:
        club = request.args.get('club')
        res = get_best_team(club, send_tactic, 'club')
    
    print(res, type(res))
    response.status = status.HTTP_200_OK
    response.data = json.dumps({'result': res})
    
    return response



####-----------------------------------------------------------------------

@app.route('/feature4', methods=['POST'])
def feature4():
    initial_overall = 80
    if 'initial_overall' in request.args:
        initial_overall = int(request.args.get('initial_overall'))
    


    # Load players data from S3
    bucket_name = os.environ.get('S3_BUCKET_NAME')
    players_file_key = "Files/players_22.csv"
    df = download_data_from_s3(bucket_name, players_file_key)
    
    df.dob = pd.to_datetime(df.dob)
    df.loc[:, 'main_position'] = df['player_positions'].apply(lambda x: x.split(',')[0])

    # Calculate overall difference
    df['overall_diff'] = df['potential'] - df['overall']
    
    # Filter players based on initial overall threshold
    filtered_data = df[df['overall'] == initial_overall]
    
    # Sort by overall difference and potential
    filtered_data.sort_values(by=['overall_diff', 'potential'], ascending=[False, False], inplace=True)
    
    # Select the top 10 players
    top_players = filtered_data.head(10)

    # Convert to JSON response format
    result = []
    for idx, player in top_players.iterrows():
        result.append({
            'position': player['main_position'],
            'Name': player['short_name'],
            'club_name': player['club_name'],
            'Faceurl': player['player_face_url'],
            'Overall': int(player['overall']),
            'Potential': int(player['potential']),
            'Age': int(player['age']),
            'height_cm': float(player['height_cm']),
            'weight_kg': float(player['weight_kg'])
        })
    
    response = Response(mimetype='application/json')
    response.status = 200
    response.data = json.dumps({'result': result})

    return response





if __name__ == "main":
    app.run()