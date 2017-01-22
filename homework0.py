import pandas as pd
import csv
import requests

CSV_URL = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'

with requests.Session() as s:
    download = s.get(CSV_URL)

    decoded_content = download.content.decode('utf-8')

    cr = csv.reader(decoded_content.splitlines(), delimiter=';')
    output_file = open('winequality_red_rawdata.csv', "w", newline="")
    writer = csv.writer(output_file, delimiter=';')
    for row in cr:
        writer.writerow(row)
    output_file.close()

df = pd.read_csv('winequality_red_rawdata.csv', sep=';')

total_rows=len(df.axes[0])
train_count=1000
test_count=total_rows-train_count

# Set a default value to class
df['class'] = 'low'
df.loc[df['quality'] <= 5,'class'] = 'low'
df.loc[df['quality'] > 5, 'class'] = 'high'

# Set a default value to sugar
df['sugar_level'] = '<= 1.5'
df.loc[df['residual sugar'] <= 1.5,'Sugar Level'] = '<= 1.5'
df.loc[(df['residual sugar'] > 1.5) & (df['residual sugar'] <= 2), 'sugar_level'] = '1.5 - 2.0'
df.loc[(df['residual sugar'] > 2) & (df['residual sugar'] <= 2.5), 'sugar_level'] = '2.0 - 2.5'
df.loc[(df['residual sugar'] > 2.5) & (df['residual sugar'] <= 3), 'sugar_level'] = '2.5 - 3.0'
df.loc[df['residual sugar'] > 3,'sugar_level'] = '> 3.0'

# Set a default value to pH
df['ph_level'] = '2.74 - 3.00'
df.loc[df['pH'] <= 3,'ph_level'] = '2.74 - 3.00'
df.loc[(df['pH'] > 3) & (df['pH'] <= 3.25), 'ph_level'] = '3.00 - 3.25'
df.loc[(df['pH'] > 3.25) & (df['pH'] <= 3.5), 'ph_level'] = '3.25 - 3.50'
df.loc[(df['pH'] > 3.5) & (df['pH'] <= 3.75), 'ph_level'] = '3.50 - 3.75'
df.loc[df['pH'] > 3.75,'ph_level'] = '3.50 - 4.01'

# Set a default value to alcohol
df['alcohol_level'] = '<= 9.0'
df.loc[df['alcohol'] <= 9,'alcohol_level'] = '<= 9.0'
df.loc[(df['alcohol'] > 9) & (df['alcohol'] <= 9.5), 'alcohol_level'] = '9.0 - 9.5'
df.loc[(df['alcohol'] > 9.5) & (df['alcohol'] <= 10), 'alcohol_level'] = '9.5 - 10.0'
df.loc[(df['alcohol'] > 10) & (df['alcohol'] <= 10.5), 'alcohol_level'] = '10.0 - 10.5'
df.loc[(df['alcohol'] > 10.5) & (df['alcohol'] <= 11), 'alcohol_level'] = '10.5 - 11.0'
df.loc[(df['alcohol'] > 11) & (df['alcohol'] <= 11.5), 'alcohol_level'] = '11.0 - 11.5'
df.loc[(df['alcohol'] > 11.5) & (df['alcohol'] <= 12), 'alcohol_level'] = '11.5 - 12.0'
df.loc[df['alcohol'] > 12,'alcohol_level'] = '> 12.0'

cols_to_keep = ['sugar_level','ph_level','alcohol_level', 'fixed acidity','volatile acidity','citric acid','chlorides','free sulfur dioxide','total sulfur dioxide','density','sulphates','class']
df[cols_to_keep].to_csv("winequality_red_formatted.csv", index=False)

df_train = pd.read_csv("winequality_red_formatted.csv", nrows=train_count)
df_test = pd.read_csv("winequality_red_formatted.csv", skiprows=range(1,train_count+1), nrows=test_count)

f1 = open("winequality_red_train.arff","w")
f2 = open("winequality_red_test.arff","w")

arffList = []
arffList.append("@relation " + "winequality_white" + "\n")

arffList.append("@attribute " + "sugar_level" + " {'<= 1.5', '1.5 - 2.0', '2.5 - 3.0', '2.0 - 2.5', '> 3.0'}" + "\n")
arffList.append("@attribute " + "ph_level" + " {'2.74 - 3.00', '3.00 - 3.25', '3.25 - 3.50', '3.50 - 3.75', '3.50 - 4.01'}" + "\n")
arffList.append("@attribute " + "alcohol_level" + " {'<= 9.0', '9.0 - 9.5', '9.5 - 10.0', '10.0 - 10.5', '10.5 - 11.0', '11.0 - 11.5', '11.5 - 12.0', '> 12.0'}" + "\n")
arffList.append("@attribute " + "fixed_acidity" + " numeric" + "\n")
arffList.append("@attribute " + "volatile_acidity" + " numeric" + "\n")
arffList.append("@attribute " + "citric_acid" + " numeric" + "\n")
arffList.append("@attribute " + "chlorides" + " numeric" + "\n")
arffList.append("@attribute " + "free_sulfur_dioxide" + " numeric" + "\n")
arffList.append("@attribute " + "total_sulfur_dioxide" + " numeric" + "\n")
arffList.append("@attribute " + "density" + " numeric" + "\n")
arffList.append("@attribute " + "sulphates" + " numeric" + "\n")
arffList.append("@attribute " + "class" + " {high, low}" + "\n\n")

arffList.append("@data " + "\n\n")

f1.writelines(arffList)
df_train.to_csv(f1, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)

f2.writelines(arffList)
df_test.to_csv(f2, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)