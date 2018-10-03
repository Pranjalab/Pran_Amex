# Pran_Amex
This is 2018 American Express Campus Analyze Thisis a first-of-its-kind data analytics competition by American Express.

#Background

The students of Ivyleague University have decided to embark on a start-up journey. They decided to establish and operate a departmental store named Tabz Departmental Store. This store offers a wide variety of products and services that caters to the daily needs of all the people staying in the city in which the university is located. Apart from Tabz, there are a couple of other departmental stores in the city. In order to tackle the intense competition, Tabz decided to offer its products and services on credit i.e., customers can consume products and services without paying for it at the time of consumption. Payment for the products and services consumed from Tabz during the month can be made at the end of the month. This is a revolutionary idea when compared to the traditional mode of cash payment and is expected to simplify the lives of people in the city while giving competitive advantage to Tabz. So, the founders decided to launch a co-branded credit card in association with a local bank, Banco. Tabz has decided to offer two types of cards:

    Charge card: The balance is required to be paid in full each month
    Lending card: Lending cards allow the customer to pay the balance over a period of time subject to interest being charged

An individual can apply for only one of the two types of credit card on offer. In order to extend the credit card to the individuals, Banco must first underwrite the applicant. Underwriting is the process by which the lender decides whether an applicant is creditworthy and should receive a credit line. Given the innovative business model of Tabz and the sound reputation of its founders, thousands of residents in the city submitted their application forms for the co-branded credit card from Tabz. Along with the data present in application forms, Banco also has access to the consumer bureau. Bureau is an agency that aggregates consumer borrowing and payment information for the purpose of assessing credit-worthiness of an individual and setting a limit on the cumulative credit that can be extended to an individual by lenders.

Banco has hired you to help underwrite each applicant and predict the credit worthiness of an individual. Banco has provided you with the customer application and bureau data with the default tagging i.e., if a customer has missed cumulative of 3 payments across all open trades, his default indicator is 1 else 0. Data consists of independent variables at the time T0 and the actual performance of the individual (Default/ Non Default) after 12 months i.e., at time T12. Banco’s expectation from you is to predict if an applicant will go default in next 12 months from the time of application submission.

At the time of launch of credit card, Banco earmarked a budget of $ 50,000 for processing the applications. After 12 months, Banco is keen to know if it had processed the right applications. If you predicted that the applicant will not default and the applicant actually does not default after 12 months, then Banco assigns a retrospective processing charge of $5. In every other case, Banco assigns a charge of $ 10.
Problem Statement

You have to create a list of applications in the order in which Banco should process them. With an objective to maintain healthy financials, Banco would like to process least risky applications first. Against each application, you also have to provide your prediction of default - 1 or 0, where 1 indicates a default and 0 indicates no default.

Assume:

    A resident of the city can submit only a single application form
    None of the applications submitted are fraudulent
    State any other assumptions in your final submission

Data for Analysis

Following files can be downloaded for your analysis.

    Training_dataset.csv: This dataset contains:
        Applicant level historic credit history
        Performance in terms of default tagging i.e. 1 for default and 0 for no default
        Application and bureau data
    Leaderboard_dataset.csv: This data has historical applicantlevel data along with all the variables in the training dataset. The actual performance i.e. default tagging is not present in this data.
    Evaluation_dataset.csv: This data has applicant level data along with all the variables in the training dataset. The actual performance i.e. default tagging is not present in this data.
    Data_Dictionary.xlsx: This sheet will give you the description of all the variables contained in the 3 datasets above.
# Solution
In order to archive accurate solution i have tried different methods from which XGBoost perform best following are other method which are used:
    
    Artificial neural network: Amex_ANN.py
    AdaBoostClassifier : Amex_ABC.py
    GradientBoostingClassifier: Amex_GBC.py
    RandomForestClassifier: Amex_RFC.py
    XGBoost: Amex.py and Amex_18.py
    
    LinearDiscriminantAnalysis: For Feature engineering (In most of the files except XGBoost)

refer to the Data folder for more Info about Problem statement 
