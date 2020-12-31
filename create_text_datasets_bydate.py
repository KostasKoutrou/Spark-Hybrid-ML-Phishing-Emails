import SpyderTest, os, mailbox, utils, re, nltk
#nltk.download('punkt')


def append_file(filepath, msg):
    if(len(msg)>0):
        f = open(filepath, "a")
        f.write(msg)
        f.write("\n")
        f.close()

for root, dirs, files in os.walk("phishing_datasets"):
    pass

for file in files:
    mbox = mailbox.mbox(root+"/"+file, factory=SpyderTest.mbox_reader)
    print("Processing file: " + root+"/"+file)
    for message in mbox:
        if(not utils.is_empty(message)):
            clean_text = utils.get_clean_text(message)
#            clean_text = clean_text.replace("\n"," ")
#            clean_text = clean_text.replace("\t"," ")
#            clean_text = re.sub(' +',' ',clean_text)
            nltk_tokens = nltk.word_tokenize(clean_text)
            clean_text = ""
            for term in nltk_tokens:
                if("'" not in term):
                    clean_text += " "
                clean_text += term
#            clean_text = " ".join(nltk_tokens)
            if(SpyderTest.year().get_feat(message)=="2015"): append_file("phishing_date_text/phishing2015_text.txt", clean_text)
            elif(SpyderTest.year().get_feat(message)=="2016"): append_file("phishing_date_text/phishing2016_text.txt", clean_text)
            elif(SpyderTest.year().get_feat(message)=="2017"): append_file("phishing_date_text/phishing2017_text.txt", clean_text)
            elif(SpyderTest.year().get_feat(message)=="2018"): append_file("phishing_date_text/phishing2018_text.txt", clean_text)
            elif(SpyderTest.year().get_feat(message)=="2019"): append_file("phishing_date_text/phishing2019_text.txt", clean_text)