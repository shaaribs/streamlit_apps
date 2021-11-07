import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle


penguin_df = pd.read_csv("penguins.csv")
#remove invalid data (nan value) guna dropna
penguin_df.dropna(inplace=True)

output = penguin_df['species'] # target
features = penguin_df[['island','bill_length_mm','bill_depth_mm',
'flipper_length_mm','body_mass_g','sex']]

#print(output.tail())

#original after cleaning Nan Values
#print(features.tail())

#features after encoding
features = pd.get_dummies(features)

output, uniques = pd.factorize(output)
# train test split
x_train, x_test, y_train, y_test = train_test_split(features, output, test_size=0.8)
rfc = RandomForestClassifier(random_state=15)
rfc.fit(x_train, y_train)


print('Successfull!')

y_pred = rfc.predict(x_test)
score = accuracy_score(y_pred, y_test)
print("Our accuracy score for this model is {}".format(score))

# save the penguin RF classifier
rfc_pickle = open("random_forest_penguin.pickle", 'wb')
pickle.dump(rfc, rfc_pickle)
rfc_pickle.close()

output_pickle = open('output_penguin.pickle','wb') #write bytes
pickle.dump(uniques, output_pickle)
output_pickle.close()

#print('Ok')
#print()
#print(features.tail())

#print(penguin_df.head())
#print(penguin_df.tail())
