from firebase import firebase
import hashlib
url = "https://sales-system-project.firebaseio.com/"
fb = firebase.FirebaseApplication(url, None)

def upload_words(words,account):
    count_words = fb.get("/意見區域/"+account,None)
    print(count_words)
    fb.post('/意見區域/'+account, data = words )
