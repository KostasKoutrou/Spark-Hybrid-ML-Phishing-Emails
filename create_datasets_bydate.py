import SpyderTest, os, mailbox


def append_file(filepath, msg):
    f = open(filepath, "a")
    f.write(msg.as_string())
    f.write("\n")
    f.close()

for root, dirs, files in os.walk("phishing_datasets"):
    pass

mbox2015 = mailbox.mbox("phishing_date/phishing2015.mbox")#, factory=SpyderTest.mbox_reader)
mbox2016 = mailbox.mbox("phishing_date/phishing2016.mbox")#, factory=SpyderTest.mbox_reader)
mbox2017 = mailbox.mbox("phishing_date/phishing2017.mbox")#, factory=SpyderTest.mbox_reader)
mbox2018 = mailbox.mbox("phishing_date/phishing2018.mbox")#, factory=SpyderTest.mbox_reader)
mbox2019 = mailbox.mbox("phishing_date/phishing2019.mbox")#, factory=SpyderTest.mbox_reader)
mbox2020 = mailbox.mbox("phishing_date/phishing2020.mbox")#, factory=SpyderTest.mbox_reader)

#mbox2015path = "phishing_date/phishing2015.mbox"
#mbox2016path = "phishing_date/phishing2016.mbox"
#mbox2017path = "phishing_date/phishing2017.mbox"
#mbox2018path = "phishing_date/phishing2018.mbox"
#mbox2019path = "phishing_date/phishing2019.mbox"
#mbox2020path = "phishing_date/phishing2020.mbox"

for file in files:
    mbox = mailbox.mbox(root+"/"+file)#, factory=SpyderTest.mbox_reader)
    print("Processing file: " + root+"/"+file)
    for message in mbox:
        if(SpyderTest.year().get_feat(message)=="2015"): mbox2015.add(message)
        elif(SpyderTest.year().get_feat(message)=="2016"): mbox2016.add(message)
        elif(SpyderTest.year().get_feat(message)=="2017"): mbox2017.add(message)
        elif(SpyderTest.year().get_feat(message)=="2018"): mbox2018.add(message)
        elif(SpyderTest.year().get_feat(message)=="2019"): mbox2019.add(message)
        elif(SpyderTest.year().get_feat(message)=="2020"): mbox2020.add(message)
#        if(SpyderTest.year().get_feat(message)=="2015"): append_file(mbox2015path, message)
#        elif(SpyderTest.year().get_feat(message)=="2016"): append_file(mbox2016path, message)
#        elif(SpyderTest.year().get_feat(message)=="2017"): append_file(mbox2017path, message)
#        elif(SpyderTest.year().get_feat(message)=="2018"): append_file(mbox2018path, message)
#        elif(SpyderTest.year().get_feat(message)=="2019"): append_file(mbox2019path, message)
#        elif(SpyderTest.year().get_feat(message)=="2020"): append_file(mbox2020path, message)
