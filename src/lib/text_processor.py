import nltk
import re

class TextProcessor:
    def __init__(self, labels):
        self.labels = labels
        self.stemmer = nltk.stem.snowball.SnowballStemmer("danish")
        self.stopwords = nltk.corpus.stopwords.words('danish')

    def remove_stopwords(self, text):
        return str([w for w in text.split(" ") if not w in self.stopwords])

    def list_to_string(self, s):
        new = ""
        for x in s:
            new += x
        return new

    def filter_text(self, text):
        text = self.remove_stopwords(text)
        text = re.findall("[a-z, A-Z, 0-9, :]", text)
        text = self.list_to_string(text)
        return text

    def clean_labels(self):
        self.fill_subjects()
        self.fill_omlidt()
        self.fill_location()
        return self.labels

    def fill_subjects(self):
        # Fill missing subjects
        dictio = {}

        i = 0
        for row in self.labels.iterrows():
            current_title = row[1]['Story title']
            current_subject = self.strip_special_chars(self.labels.loc[i]['Story subject']) if self.labels.loc[i]['Story subject'] is not None else None

            if current_title in dictio:
                self.labels.loc[i]['Story subject'] = dictio[current_title]
            else:
                if self.labels.loc[i]['Story subject'] is not None:
                    dictio[current_title] = current_subject
            i += 1

        i = 0
        for row in self.labels.iterrows():
            current_title = row[1]['Story title']
            current_subject = self.labels.loc[i]['Story subject']
            if current_title in dictio:
                self.labels.loc[i]['Story subject'] = dictio[current_title]
            else:
                if self.labels.loc[i]['Story subject'] is not None:
                    dictio[current_title] = current_subject
            i += 1

    def fill_omlidt(self):
        omlidt = self.labels[self.labels['Om lidt'] == 'True']
        for i, row in omlidt.iterrows():
            title = row['Story title']
            current_title = row['Story title']
            frame = row['Frame']

            c = 0
            while title == current_title:
                self.labels.loc[i-c]['Om lidt'] = 'True'
                c += 1
                current_title = self.labels.loc[i-c]['Story title']
                current_title = self.labels.loc[i-c]['Story title']

            current_title = row['Story title']
            c = 0
            while title == current_title:
                self.labels.loc[i+c]['Om lidt'] = 'True'
                c += 1
                current_title = self.labels.loc[i+c]['Story title']
                current_title = self.labels.loc[i+c]['Story title']

    def fill_location(self):
        direkte = self.labels[self.labels['Direkte'] == 'True']
        for i, row in direkte.iterrows():
            location = self.strip_special_chars(row['Location']) if row['Location'] is not None else None
            frame = row['Frame']
            direkte_ = True


            if location is not None:
                c = 0
                while direkte_:
                    self.labels.loc[i-c]['Location'] = location
                    c += 1
                    direkte_ = self.labels.loc[i-c]['Direkte']


                direkte_ = True
                c = 0
                while direkte_:
                    self.labels.loc[i+c]['Location'] = location
                    c += 1
                    direkte_ = self.labels.loc[i+c]['Direkte']



    def strip_special_chars(self, text):
        text = text.strip()
        text = text.lstrip('.').rstrip('.')
        text = text.lstrip("'").rstrip("''")
        text = text.lstrip('|').rstrip('|')
        text = text.lstrip('-').rstrip('-')
        return text
