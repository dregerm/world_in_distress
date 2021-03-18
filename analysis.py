import re
import os
import csv
import json
import numpy as np
from scipy.stats import spearmanr
from collections import defaultdict
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score,precision_score
import math



#getting data from csvs....

names = []

with open('./data/co2_emissions.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter = ',')
    for row in csv_reader:
        names.append(row[2].lower())


names.sort()
countries_matrix = [[] for x in range(len(names))]


for i in range(len(names)):
    countries_matrix[i].append(names[i])


#Series Name,Series Code,Country Name,Country Code,2005 [YR2005],2006 [YR2006],2007 [YR2007],2008 [YR2008],2009 [YR2009],2010 [YR2010],2011 [YR2011],2012 [YR2012],2013 [YR2013],2014 [YR2014],2015 [YR2015],2016 [YR2016],2017 [YR2017],2018 [YR2018],2019 [YR2019],2020 [YR2020]
with open('./data/arable_land.csv') as arable_land:
    csv_reader = csv.reader(arable_land, delimiter = ',')
    
        #row[2] is the name of area for given data....
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
                
            countries_matrix[names.index(row[2].lower())].append(data_list) 

#this below code makes sure that something is added to countries_matrix
#in the spot of where data value should go if country name not in the dataset.
# if do not have this,the country's data will be offset because of the fact that 
# the index of other countries data will be different. 
for item in countries_matrix:
    if len(item) ==  1:
        item.append([])



#Series Name,Series Code,Country Name,Country Code,2005 [YR2005],2006 [YR2006],2007 [YR2007],2008 [YR2008],2009 [YR2009],2010 [YR2010],2011 [YR2011],2012 [YR2012],2013 [YR2013],2014 [YR2014],2015 [YR2015],2016 [YR2016],2017 [YR2017],2018 [YR2018],2019 [YR2019],2020 [YR2020]
with open('./data/combustable_renewables_and_waste_%nrg.csv') as combust_renews_nrg:
    csv_reader = csv.reader(combust_renews_nrg, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 

for item in countries_matrix:
    if len(item) ==  2:
        item.append([])
#Series Name,Series Code,Country Name,Country Code,2005 [YR2005],2006 [YR2006],2007 [YR2007],2008 [YR2008],2009 [YR2009],2010 [YR2010],2011 [YR2011],2012 [YR2012],2013 [YR2013],2014 [YR2014],2015 [YR2015],2016 [YR2016],2017 [YR2017],2018 [YR2018],2019 [YR2019],2020 [YR2020]
with open('./data/fossil_fuel_%total.csv') as fossil_fuel_percent:
    csv_reader = csv.reader(fossil_fuel_percent, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 

for item in countries_matrix:
    if len(item) ==  3:
        item.append([])

#Year,Country,Total,Solid Fuel,Liquid Fuel,Gas Fuel,Cement,Gas Flaring,Per Capita,Bunker fuels (Not in Total)
with open('./data/fossil-fuel-co2-emissions-by-nation_csv.csv') as co2_from_fossils:
    csv_reader = csv.reader(co2_from_fossils, delimiter = ',')
    temp_list = [[] for x in range(len(names))]

    for i in range(len(names)):
        temp_list[i].append(names[i])
    
    for row in csv_reader:
        for name in names:
            if(name in row[1].lower() and int(row[0]) >= 2005):
                temp_list[names.index(name)].append(row[2])

    for item in temp_list:
        data_list = []
        for i in range(len(item)):
            if i > 0:
                data_list.append(item[i])
        
        countries_matrix[names.index(item[0])].append(data_list)
for item in countries_matrix:
    if len(item) ==  4:
        item.append([])

#Series Name,Series Code,Country Name,Country Code,2005 [YR2005],2006 [YR2006],2007 [YR2007],2008 [YR2008],2009 [YR2009],2010 [YR2010],2011 [YR2011],2012 [YR2012],2013 [YR2013],2014 [YR2014],2015 [YR2015],2016 [YR2016],2017 [YR2017],2018 [YR2018],2019 [YR2019],2020 [YR2020]
with open('./data/gdp.csv') as gdp:
    csv_reader = csv.reader(gdp, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 

for item in countries_matrix:
    if len(item) ==  5:
        item.append([])


#am using lat long list to determine only_country_names because
#this list only has strict country names... there is no 'world' lat 
#and longitude for example...
only_country_names = []

#country identifier, latitude, longitude, name	
with open('./data/lat_and_longitude.csv') as lat_and_long:
    csv_reader = csv.reader(lat_and_long, delimiter = ',')
    for row in csv_reader:
        if(row[3].lower() in names):
            only_country_names.append(row[3].lower())
            #countries_matrix[names.index(row[3].lower())].append([row[1],row[2]])

#for item in countries_matrix:
#    if len(item) ==  6:
#        item.append([])

#Series Name,Series Code,Country Name,Country Code,2005 [YR2005],2006 [YR2006],2007 [YR2007],2008 [YR2008],2009 [YR2009],2010 [YR2010],2011 [YR2011],2012 [YR2012],2013 [YR2013],2014 [YR2014],2015 [YR2015],2016 [YR2016],2017 [YR2017],2018 [YR2018],2019 [YR2019],2020 [YR2020]



#Series Name,Series Code,Country Name,Country Code,2005 [YR2005],2006 [YR2006],2007 [YR2007],2008 [YR2008],2009 [YR2009],2010 [YR2010],2011 [YR2011],2012 [YR2012],2013 [YR2013],2014 [YR2014],2015 [YR2015],2016 [YR2016],2017 [YR2017],2018 [YR2018],2019 [YR2019],2020 [YR2020]
#in kilotons
with open('./data/population.csv') as pop:
    csv_reader = csv.reader(pop, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 
    
for item in countries_matrix:
    if len(item) ==  6:
        item.append([])


with open('./data/life_expectancy.csv') as life_expectancy:
    csv_reader = csv.reader(life_expectancy, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 
    
for item in countries_matrix:
    if len(item) ==  7:
        item.append([])

with open('./data/co2_emissions.csv') as co2:
    csv_reader = csv.reader(co2, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 
    
for item in countries_matrix:
    if len(item) ==  8:
        item.append([])


with open('./data/renewable_energy_%total.csv') as renewable_energy:
    csv_reader = csv.reader(renewable_energy, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 
    
for item in countries_matrix:
    if len(item) == 9:
        item.append([])

with open('./data/agricultural_land%.csv') as agricultural_land:
    csv_reader = csv.reader(agricultural_land, delimiter = ',')
    for row in csv_reader:
        if(row[2].lower() in names):
            data_list = []
            for i in range(len(row)):
                if i > 3:
                    data_list.append(row[i])
            countries_matrix[names.index(row[2].lower())].append(data_list) 
    
for item in countries_matrix:
    if len(item) == 10:
        item.append([])

#---------------------------------------------------------
#>>> analysis



#this function will clean all columns based on if a '..' exists 
#within column_index
def clean_countries_on_column(column_index,matrix):
    #find indexes_to_remove
    for item in matrix:
        indexes_to_remove = []
        for i in range(len(item[column_index])):
            if item[column_index][i] == '..':
                indexes_to_remove.append(i)

    #now delete data that is in index of indexes_to_remove...
        for i in range(len(item)):
        #this below if statement just makes sure does not change some
        #datasets that is not year dependent
            if(i != 0):
                temp_list = []
                for j in range(len(item[i])):
                    if j not in indexes_to_remove:
                        temp_list.append(item[i][j])
                item[i] = temp_list

#makes a copy of 3d matrix
def make_copy(three_d_matrix):
    copy = []
    for item in three_d_matrix:
        copy_minor = []
        for value_list in item:
            copy_minor.append(value_list[:])
        copy.append(copy_minor)
    return copy

#finds correlation between two column indexes for given country
def find_correlation(index1, index2, country):
    new_matrix = make_copy(countries_matrix)
    clean_countries_on_column(index1,new_matrix)
    clean_countries_on_column(index2,new_matrix)
    for item in new_matrix:
        if item[0] == country:
            
            if(len(item[index1]) > 1 and len(item[index2]) > 1):
                #making sure item[index1] and item[index2] are not constant arrays.
                continue_on = False
                continue_on2 = False
                first_val = item[index1][0]
                for value in item[index1]:
                    if value != first_val:
                        continue_on = True
                first_val = item[index2][0]
                for value in item[index2]:
                    if value != first_val:
                        continue_on2 = True
                if(continue_on and continue_on2):
                    correlation,_ = spearmanr(item[index1],item[index2])
                    return (correlation)
                else: return None

            else: return None

#divides country names into 'number_of_splits' splits that go from lowest gdp-highest gdp


#returns the total gdp of a country
def total_gdp(country_name):
    gdp_column = 5
    new_matrix = make_copy(countries_matrix)
    clean_countries_on_column(gdp_column,new_matrix)
    for country in new_matrix:
        if country[0] == country_name:
            total_gdp = 0
            for value in country[gdp_column]:
                total_gdp += int(float(value))
    return total_gdp


    # % number_of_splits
    # if 0... number_of_splits - 1
    
#returns the mean of input_list
def mean(input_list):
    return sum(input_list)/len(input_list)

#says meaning of column number for value in countries_matrix
def column_to_meaning(column):
    if column == 0:
        return 'name'
    if column == 1:
        return 'arable_land %'
    if column == 2:
        return 'combustable_renewables_and_waste_%_nrg'
    if column == 3:
        return 'fossil_fuel_%_total_nrg'
    if column == 4:
        return 'fossil_fuel_co2_by_nation'
    if column == 5:
        return 'gdp'
    if column == 6:
        return 'population'
    if column == 7:
        return 'life_expectancy'
    if column == 8:
        return 'co2'
    if column == 9:
        return 'renewable_nrg_%_of_total'
    if column == 10:
        return 'agricultural_land_%'

#divide countrie names based on gdp from lowest-highest
def divide_across_gdp(number_of_splits):
    all_gdps = []     
    gdp_column = 5
    value_pairing = defaultdict(lambda: [])
    new_matrix = make_copy(countries_matrix)
    clean_countries_on_column(gdp_column,new_matrix)
    for country in new_matrix:
        if(len(country[gdp_column]) > 0):
            gdp = mean(clean_up_value_list(country[gdp_column]))
        else:
            gdp = 0

        all_gdps.append(gdp)
        value_pairing[gdp].append(country[0])
    
    all_gdps.sort()

    num_splits = []

    

    n = int(len(all_gdps)/number_of_splits)
    for i in range(0, len(all_gdps), n):
        num_splits.append(all_gdps[i:i + n])

    name_splits = [[] for i in range(len(num_splits))]
    for i in range(len(num_splits)):
        index = 0
        for j in range(len(num_splits[i])):
            if len(value_pairing[num_splits[i][j]]) > 1:
                country_gdps = value_pairing[num_splits[i][j]]
                country_name = country_gdps[index][0]
                index += 1
            else:
                index = 0
                country_gdps = value_pairing[num_splits[i][j]]
                country_name = country_gdps[index]
            name_splits[i].append(country_name)
    return name_splits

#shows correlation between said column and co2 according to gdp level
def show_correlation_plots(column,ymax):
    correlations = [];
    divided_by_gdp = divide_across_gdp(3)

    for country in divided_by_gdp[2]:
        correlation = find_correlation(column,8,country)
        if correlation != None:
            correlations.append(correlation)

    plt.figure(figsize=(8,3))
    plt.subplot(1,3,1)
    plt.title('high gdp countries')
    plt.xlabel('correlation ' + column_to_meaning(column) + '/co2')
    plt.ylabel('quantity')
    plt.axvline(mean(correlations), color='k', linestyle='dashed', linewidth=1.5)
    plt.ylim(0,ymax)
    plt.xlim((-1,1))
    plt.hist(correlations)

    correlations = [];
    for country in divided_by_gdp[1]:
        correlation = find_correlation(column,8,country)
        if correlation != None:
            correlations.append(correlation)

    plt.subplot(1,3,2)
    plt.title('medium gdp countries')
    plt.xlabel('correlation ' + column_to_meaning(column) + '/co2')
    plt.axvline(mean(correlations), color='k', linestyle='dashed', linewidth=1.5)
    plt.ylim(0,ymax)
    plt.xlim((-1,1))
    plt.hist(correlations)

    correlations = [];
    for country in divided_by_gdp[0]:
        correlation = find_correlation(column,8,country)
        if correlation != None:
            correlations.append(correlation)

    plt.subplot(1,3,3)
    plt.title('low gdp countries')
    plt.xlabel('correlation ' + column_to_meaning(column) + '/co2')
    plt.axvline(mean(correlations), color='k', linestyle='dashed', linewidth=1.5)
    plt.ylim(0,ymax)
    plt.xlim((-1,1))
    plt.hist(correlations)
    

    plt.show()

#deletes countries that have '..'s for all of a certain type of data
def delete_countries_only_dots(matrix):
    new_matrix = make_copy(matrix)

    return_matrix = []
    for country in new_matrix:
        continue_on = []
        for value_list in country:
            list_is_good = False
            for value in value_list:
                if value != '..':
                    list_is_good = True
                if list_is_good:
                    continue_on.append(0)
                else:
                    continue_on.append(1)
            if len(value_list) == 0:
                continue_on.append(1)

        if 1 not in continue_on:
            return_matrix.append(country)

    return return_matrix

#returns list without any '..'s
def clean_up_value_list(value_list):
    return_list = []
    for value in value_list:
        if value != '..':
            return_list.append(float(value))
    return return_list

#averages out all lists within matrix (except value at index 0)
def average_columns(matrix):
    new_matrix = make_copy(matrix)
    averaged_matrix = []
    for country in new_matrix:
        return_country_list = []
        for value_list in country:
            if value_list == country[0]:
                return_country_list.append(country[0])
            else:
                return_country_list.append(mean(clean_up_value_list(value_list)))
        averaged_matrix.append(return_country_list)
        
    return averaged_matrix

#creates level 0-10 for what a country average is in certain column
def column_value_levels(column, matrix):
    
    all_values = []
    all_splits = []
    value_pairing = defaultdict(lambda: [])

    for country in matrix:
        #print(country)
        all_values.append(country[column])
        
        value_pairing[country[column]].append(country)

    

    all_values.sort()

    n = int(len(all_values)/10)
    for i in range(0,len(all_values), n):
        all_splits.append(all_values[i:i + n])
    
    

    for i in range(len(all_splits)):
        for j in range(len(all_splits[i])):
            
            index = 0
            
            if (len(value_pairing[all_splits[i][j]]) > 1):
                countrys = value_pairing[all_splits[i][j]]
                country_list = countrys[index]
                index += 1
            else:
                index = 0
                countrys = value_pairing[all_splits[i][j]]

                country_list = countrys[index]
               
            for country in matrix:
                if country[0] == country_list[0]:
                    country[column] = i

#makes every column on a scale from 0-10
def create_column_value_levels(matrix):
    new_matrix = average_columns(delete_countries_only_dots(matrix))
    
    for i in range(11):
        if i != 0:
            column_value_levels(i,new_matrix)
    
    return new_matrix

def sigmoid(x):
      return 1 / (1 + math.exp(-x))

def name_to_list(country_name,matrix):
    for item in matrix:
        if item[0] == country_name:
            return item


        




#within countries_matrix...
#column 0 = name
#column 1 = arable_land %
#column 2 = combustable_renewables_and_waste_%nrg
#column 3 = fossil_fuel_% of total erg
#column 4 = fossil-fuel-co2-emissions-by-nation_csv
#column 5 = gdp
#column 6 = population
#column 7 = life expectancy
#column 8 = total co2 per year from 2005 in millions of tons
#column 9 = renewable_energy_%_of_total
#column 10 = agricultural_land_%

show_correlation_plots(6,40)

#show_correlation_plots(1,17)
#would interesting to determine which mean differences are statistiaclly significant


#test_matrix = create_column_value_levels(countries_matrix)
#for country in test_matrix:
    #if country[0] == 'united states':
        #print(country)
        
        

#OKAY now every country is a set number...

#did add total across years earlier because could just filter two columns. problem here is that
#I need to have every column filtered which would not work because would be left with very little data...
#so what I am doing is averaging each column and then am splitting each countries value into a grouping.




#so i need the target array which is co2 levels..
#and I need training array which is all the data...
#so first lets make target aray...

#clean_countries_on_column(10,new_array)




#can make the model on the 
model = MultinomialNB()
new_array = create_column_value_levels(countries_matrix)


#so for starters lets just use 5/6 of data as training data
target_test_data = []
testing_data = []
training_data = []
target_array = []
for i in range(len(new_array)):
    if i % 6 == 0:
        array_to_add = []
        for j in range(1,len(new_array[i])):
            if j != 8:
                array_to_add.append(new_array[i][j])
        
        testing_data.append(array_to_add)
        target_test_data.append(new_array[i][8])

    else:
        array_to_add = []
        for j in range(1,len(new_array[i])):
            if j != 8:
                array_to_add.append(new_array[i][j])
        target_array.append(new_array[i][8])
        training_data.append(array_to_add)
        


model.fit(training_data, target_array)
print()
print('ALL DATA')
prediction = model.predict(testing_data)
print('prediction: ' + str(prediction))
print('actual:    '+ str(target_test_data))

precision = precision_score(target_test_data, prediction,average='macro',zero_division=1)
recall = recall_score(target_test_data,prediction,average='macro', zero_division=1)

print('recall score: ' + str(recall))
print('precision score: ' + str(precision))
f1 = 2 * (precision * recall) / (precision + recall)
print('f1 score: ' + str(f1))


#model is fit on all GDP ranges... what if I split up the GDPs though??

names = divide_across_gdp(2)
new_array = create_column_value_levels(countries_matrix)
high_gdps = names[1]
low_gdps = names[0]

high_gdps_list = []
medium_gdps_list = []
low_gdps_list = []

for name in high_gdps:
    country_to_add = name_to_list(name,new_array)
    if country_to_add:
        high_gdps_list.append(country_to_add)

for name in low_gdps:
    country_to_add = name_to_list(name,new_array)
    if country_to_add:
        low_gdps_list.append(country_to_add)
        

#creating the testing and training data...


#FOR high GDP
print()
print("HIGH GDP MODEL")
target_test_data = []
testing_data = []
training_data = []
target_array = []



for i in range(len(high_gdps_list)):
    if i % 6 == 0:
        array_to_add = []
        for j in range(1,len(high_gdps_list[i])):
            if j != 8:
                array_to_add.append(high_gdps_list[i][j])
        
        testing_data.append(array_to_add)
        target_test_data.append(high_gdps_list[i][8])

    else:
        array_to_add = []
        for j in range(1,len(high_gdps_list[i])):
            if j != 8:
                array_to_add.append(high_gdps_list[i][j])
        target_array.append(high_gdps_list[i][8])
        training_data.append(array_to_add)
        
model = MultinomialNB()
model.fit(training_data, target_array)
prediction = model.predict(testing_data)
print('prediction: ' + str(prediction))
print('actual:    '+ str(target_test_data))

precision = precision_score(target_test_data, prediction,average='macro',zero_division=1)
recall = recall_score(target_test_data,prediction,average='macro', zero_division=1)

print('recall score: ' + str(recall))
print('precision score: ' + str(precision))
f1 = 2 * (precision * recall) / (precision + recall)
print('f1 score: ' + str(f1))


#low
print()
print("LOW GDP MODEL")
target_test_data = []
testing_data = []
training_data = []
target_array = []



for i in range(len(low_gdps_list)):
    if i % 6 == 0:
        array_to_add = []
        for j in range(1,len(low_gdps_list[i])):
            if j != 8:
                array_to_add.append(low_gdps_list[i][j])
        
        testing_data.append(array_to_add)
        target_test_data.append(low_gdps_list[i][8])

    else:
        array_to_add = []
        for j in range(1,len(low_gdps_list[i])):
            if j != 8:
                array_to_add.append(low_gdps_list[i][j])
        target_array.append(low_gdps_list[i][8])
        training_data.append(array_to_add)
        
model = MultinomialNB()
model.fit(training_data, target_array)
prediction = model.predict(testing_data)
print('prediction: ' + str(prediction))
print('actual:    '+ str(target_test_data))

precision = precision_score(target_test_data, prediction,average='macro',zero_division=1)
recall = recall_score(target_test_data,prediction,average='macro', zero_division=1)

print('recall score: ' + str(recall))
print('precision score: ' + str(precision))
f1 = 2 * (precision * recall) / (precision + recall)
print('f1 score: ' + str(f1))




