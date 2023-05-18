# TALENT HUNTING WITH MACHINE LEARNING ALGORITHMS

![image](https://github.com/baranylcn/Scotium-ML-Prediction/assets/98966968/fd6e2c95-1bb3-4e42-942a-50ed57fd96c4)

## Bussines Problem
Predicting which class (average, highlighted) player is according to the points given to the characteristics of the players.

## Dataset Story
The data set consists of information from Scoutium, which includes the features and scores of the football players evaluated by the scouts according to the characteristics of the footballers observed in the matches.

### Scoutium Attributes CSV File
- Total Features : 8
- Total Row : 10.730
- CSV File Size : 527 KB



| Feature | Description |
|----------|----------|
| task_response_id  | The set of a scout's assessments of all players on a team's roster in a match  |
| match_id  | The id of the match  |
| evaluator_id  | The id of the evaluator(scout)  |
| player_id  | The id of the player  |
| position_id  | The id of the position played by the relevant player in that match. 1-Goalkeeper, 2-Stopper, 3-Right-back, 4-Left-back, 5-Defensive midfielder, 6-Central midfield, 7-Right wing, 8-Left wing, 9-Attacking midfielder, 10-Striker  |
| analysis_id  | A set containing a scout's attribute evaluations of a player in a match  |
| attribute_id  | The id of each attribute the players were evaluated for  |
| attribute_value  | Value (points) given by a scout to a player's attribute  |


### Scoutium Potential Labels CSV File
- Total Features : 5
- Total Row : 322
- CSV File Size : 12 KB

| Feature | Description |
|----------|----------|
| task_response_id  | The set of a scout's assessments of all players on a team's roster in a match  |
| match_id  | The id of the match  |
| evaluator_id  | The id of the evaluator(scout)  |
| player_id  | The id of the player  |
| potential_label	  | Label showing the final decision of an observer regarding a player in the match. (target variable)  |





