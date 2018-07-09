#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print "\nData points (people) in the data set:", len(enron_data), "\n    ",
print_count = 0
for key in enron_data:
    print key, "\t\t",
    print_count += 1
    if (print_count == 4):
        print "\n    ",
        print_count = 0

print "\n\nFeatures available:", len(enron_data["SKILLING JEFFREY K"]), "\n    ",
print_count = 0
for key in enron_data["SKILLING JEFFREY K"]:
    print key, "\t\t",
    print_count += 1
    if (print_count == 4):
        print "\n    ",
        print_count = 0

poi_count = 0
for key in enron_data:
    if (enron_data[key]["poi"] == 1):
        poi_count += 1
print "\n\nPOI Count:", poi_count

print "\nTotal value of James Prentice's stock:", enron_data["PRENTICE JAMES"]["total_stock_value"]

print "\nNumber of email messages from Wesley Colwell to POI:", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]

print "\nValue of stock options exercised by Jeffrey K Skilling:", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]

print "\nLay's total payments:\t", enron_data["LAY KENNETH L"]["total_payments"]
print "Skilling's total payments:\t", enron_data["SKILLING JEFFREY K"]["total_payments"]
print "Fastow's total payments:\t", enron_data["FASTOW ANDREW S"]["total_payments"]

print "\nKenneth Lay's features:"
for key in enron_data["LAY KENNETH L"]:
    print "   ", key, ":", enron_data["LAY KENNETH L"][key]

print "\nNumber of people with quantified salary:",
count = 0
for key in enron_data:
    if (enron_data[key]["salary"] != "NaN"):
        count += 1
print count

print "\nNumber of people with a known email address:",
count = 0
for key in enron_data:
    if (enron_data[key]["email_address"] != "NaN"):
        count += 1
print count

print "\nNumber of people that have 'NaN' for total payments:",
whole_count = 0.0
count = 0.0
for key in enron_data:
    whole_count += 1.0
    if (enron_data[key]["total_payments"] == "NaN"):
        count += 1.0
print count, "which is", (100 * count / whole_count), "percent"

print "\nNumber of POI's that have 'NaN' for total payments:",
poi_count = 0.0
count = 0.0
for key in enron_data:
    if (enron_data[key]["poi"] == 0): continue
    poi_count += 1.0
    if (enron_data[key]["total_payments"] == "NaN"):
        count += 1.0
print count, "which is", (100 * count / poi_count), "percent"
