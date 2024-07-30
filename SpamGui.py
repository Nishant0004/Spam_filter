from tkinter import *
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import LancasterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

BG_COLOR="#89CFF0"
FONT_BOLD="Melvetica %d bold"

class SpamHam:
    def __init__(self):
        self.window=Tk()
        self.main_window()
        self.lstem = LancasterStemmer()
        self.tfvec=TfidfVectorizer(stop_words='english')
        self.datafile()

    def datafile(self):
        datafile = pickle.load(open("training_data.pkl","rb"))
        self.message_x = datafile["message_x"]
        self.classifier = datafile["classifier"]
    
    def main_window(self):
        self.window.title("Spam Detector")
        self.window.resizable(width=False,height=False)
        self.window.configure(width=520,height=400,bg=BG_COLOR)
        head_label=Label(self.window,bg="#FFA500",fg="#000",text="Welcome to the Project",
                         font=FONT_BOLD%(14),pady=10)
        head_label.place(relwidth=1)
        line = Label(self.window,width=200,bg="#000")
        line.place(relwidth=0.5,relx=0.25,rely=0.08,relheight=0.008)
        mid_label=Label(self.window,bg=BG_COLOR,fg="#0000FF",text="Spam Or Ham ? Message Detector",
                        font=FONT_BOLD%(18),pady=10)
        mid_label.place(relwidth=1,rely=0.12)
        self.answer=Label(self.window,bg=BG_COLOR,fg="#000",text="Please type message below.",
                        font=FONT_BOLD%(16),pady=10,wraplength=525)
        self.answer.place(relwidth=1,rely=0.30)

        self.msg_entry=Text(self.window,bg="#FFF",fg="#000",font=FONT_BOLD%(14))
        self.msg_entry.place(relwidth=1,relheight=0.4,rely=0.48)
        self.msg_entry.focus()
        check_button=Button(self.window,text="Check",font=FONT_BOLD%(12),width=8,bg="#000",fg="#FFF",
                            command=lambda: self.on_enter(None))        
        check_button.place(relx=0.40,rely=0.90,relheight=0.08,relwidth=0.20)

    def bow(self,message):
        mess_t = self.tfvec.fit(self.message_x)
        message_test=mess_t.transform(message).toarray()
        return message_test
    
    def mess(self,messages):
        message_x = []           
        for me_x in messages:
            me_x = str(me_x)
            me_x=''.join(filter(lambda mes:(mes.isalpha() or mes==" ") ,me_x))
            words = word_tokenize(me_x)
            message_x+=[' '.join([self.lstem.stem(word) for word in words])]
        return message_x

    def on_enter(self,event):
        msg=str(self.msg_entry.get("1.0","end"))
        message=self.mess([msg])
        self.answer.config(fg="#ff0000",text="Your message is : "+
                           ("spam" if self.classifier.predict(self.bow(message)).reshape(1,-1) else "ham"))

    def run(self):
        self.window.mainloop()

if __name__=="__main__":
    app = SpamHam()
    app.run()
