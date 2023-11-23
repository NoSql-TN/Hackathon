# The goal of this file is to provide some hint about how the song can be improved for this we will try to modify the parameters of the song and see if it improves the score
import pickle
import random
from python.core.prediction import preprocess_data

global loaded_model

def load_model():
    
    global loaded_model
    
    loaded_model = pickle.load(open('models/finalized_model.sav', 'rb'))

def predict_popularity(X):
    global loaded_model
    # predict the popularity
    
    prediction = loaded_model.predict(X)
    
    return prediction[0]

def amelioration(data):
    
    load_model()
    
    lyrics = data['lyrics']
    genre = data['genre']
    artist_name = data['artist_name']
    track_name = data['track_name']
    tempo = data['tempo']
    key = data['cle']
    explicit = data['explicit']
    duration = data['duration']
    score = data['score']
    
    improvement = {}
    good = {}
    
    # We need to iterate over all the parameters and try them one at a time to see if it improves the score
    # We will try to modify the tempo randomly
    
    new_tempo = tempo + random.randint(-30,30)
    if new_tempo == tempo:
        new_tempo = tempo + 1
    
    data = {"lyrics": lyrics, "genre": genre, "artist_name": artist_name, "track_name": track_name, "tempo": new_tempo, "cle": key , "explicit" : explicit, "duration" : duration}
    X = preprocess_data(data)
    
    new_score = predict_popularity(X)
    if new_score > score:
        if new_tempo > tempo:
            improvement['tempo'] = ("Increase to " + str(new_tempo), new_score)
        else:
            improvement['tempo'] = ("Decrease to " + str(new_tempo), new_score)
    else:
        if new_tempo > tempo:
            good['tempo'] = ("Increase to " + str(new_tempo), new_score)
        else:
            good['tempo'] = ("Decrease to " + str(new_tempo), new_score)
    
    # We will try to modify the key randomly
    new_key = random.randint(0,11)
    
    data = {"lyrics": lyrics, "genre": genre, "artist_name": artist_name, "track_name": track_name, "tempo": tempo, "cle": new_key , "explicit" : explicit, "duration" : duration}
    X = preprocess_data(data)
    
    new_score = predict_popularity(X)
    if new_score > score:
        improvement['key'] = ("Change to " + str(new_key), new_score)
    else:
        good['key'] = ("Change to " + str(new_key), new_score)
    
    # We will try to modify the explicit
    if explicit == 0:
        new_explicit = 1
    else:
        new_explicit = 0
    
    data = {"lyrics": lyrics, "genre": genre, "artist_name": artist_name, "track_name": track_name, "tempo": tempo, "cle": key , "explicit" : new_explicit, "duration" : duration}
    X = preprocess_data(data)
    
    new_score = predict_popularity(X)
    if new_score > score:
        if new_explicit > explicit:
            improvement['explicit'] = ("Set to Explicit", new_score)
        else:
            improvement['explicit'] = ("Set to Not Explicit", new_score)
    else:
        if new_explicit > explicit:
            good['explicit'] = ("Set to Explicit", new_score)
        else:
            good['explicit'] = ("Set to Not Explicit", new_score)
    
    # We will try to modify the duration randomly
    new_duration = duration + random.randint(-10,10)*1000
    
    data = {"lyrics": lyrics, "genre": genre, "artist_name": artist_name, "track_name": track_name, "tempo": tempo, "cle": key , "explicit" : explicit, "duration" : new_duration}
    X = preprocess_data(data)
    
    new_score = predict_popularity(X)
    if new_score > score:
        if new_duration > duration:
            improvement['duration'] = ("Increase to " + str(new_duration), new_score)
        else:
            improvement['duration'] = ("Decrease to " + str(new_duration), new_score)
    else:
        if new_duration > duration:
            good['duration'] = ("Increase to " + str(new_duration), new_score)
        else:
            good['duration'] = ("Decrease to " + str(new_duration), new_score)
        
    # We will try to modify the track_name randomly by adding a random word and deleting a random word
    list_of_words_to_try = ["love","hate","heart","life","death","live","die","cry","laugh","smile","sad","happy","good","bad","beautiful","ugly","pretty","handsome","nice","cool","hot","cold","warm","freezing","fire","water","earth","air","wind","storm","rain","sun","moon","star","sky","cloud","cloudy","clear","dark","light","bright","darkness","lightness","shadow","shine","shiny","glow","glowing","shine"]
    new_track_name = track_name.split()
    if len(new_track_name) > 1:
        bad_word = new_track_name.pop(random.randint(0,len(new_track_name)-1))
        new_track_name.append(list_of_words_to_try[random.randint(0,len(list_of_words_to_try)-1)])
    else:
        bad_word = None
        new_track_name.append(list_of_words_to_try[random.randint(0,len(list_of_words_to_try)-1)])
    
    new_track_name = " ".join(new_track_name)
    
    data = {"lyrics": lyrics, "genre": genre, "artist_name": artist_name, "track_name": new_track_name, "tempo": tempo, "cle": key , "explicit" : explicit, "duration" : duration}
    X = preprocess_data(data)
    
    new_score = predict_popularity(X)
    if new_score > score:
        if bad_word:
            improvement['track_name'] = ("Change " + bad_word + " to " + new_track_name, new_score)
        improvement['track_name'] = ("add" + new_track_name, new_score)
    else:
        if bad_word:
            good['track_name'] = ("Change " + bad_word + " to " + new_track_name, new_score)
        good['track_name'] = ("add" + new_track_name, new_score)
    
    # The artist_name is not really important so we will not try to modify it
    # We will try to modify the genre randomly
    list_of_genres_to_try = ['blues','country','hip hop','jazz','pop','reggae','rock']
    new_genre = list_of_genres_to_try[random.randint(0,len(list_of_genres_to_try)-1)]
    
    data = {"lyrics": lyrics, "genre": new_genre, "artist_name": artist_name, "track_name": track_name, "tempo": tempo, "cle": key , "explicit" : explicit, "duration" : duration}
    X = preprocess_data(data)
    
    new_score = predict_popularity(X)
    if new_score > score:
        improvement['genre'] = ("Change to " + new_genre, new_score)
    else:
        good['genre'] = ("Change to " + new_genre, new_score)
        
    # We will try to modify the lyrics randomly by adding 3 randoms words and deleting 3 randoms words
    words_to_try=["know","time","like","heart","come","away","go","life","feel","na","baby","want","leave","live","yeah","night","world","cause","right","long","fall","break","tell","think","hold","need","dream","thing","look","gon","good","mind","hear","home","eye","believe","walk","hand","lose","change","stay","say","blue","girl","little","song","sing","tear","stand","play","turn","wan","love","better","lonely","tonight","take","head","true","start","inside","face","place","fuck","remember","sweet","kiss","word","bring","nigga","day","give","hard","hurt","light","wait","forget","wish","soul","money","people","friend","year","stop","wrong","fool","woman","try","cold","today","pain","mean","make","cry","watch","arm","memory","close","shit","sleep","forever","fight","hide","care","real","black","got","ta","free","miss","rain","get","smile","talk","kill","dead","burn","grow","goodbye","touch","door","learn","tomorrow","lyric","save","line","darling","call","till","open","bitch","help","morning","lie","high","commercial","game","somebody","reason","music","sound","fear","lord","matter","apart","deep","wonder","see","begin","work","moment","best","wall","listen","keep","beat","dear","blood","roll","hell","maybe","run","understand","promise","lover","die","strong","road","ready","dance","grind","alive","shoot","wind","blow","young","street","sure","fade","write","late","follow","bout","easy","pas","spend","everybody","dark","throw","guess","heaven","truth","soon","hate","share","step","child","speak","catch","kind","body","white","party","reach","past","belong","star","death","waste","rest","pray","drink","yesterday","foot","afraid","breathe","alright","voice","devil","thank","laugh","lip","blind","pull","tire","round","room","ring","nothin","smoke","summer","crazy","build","steal","blame","hang","fine","damn","house","lead","search","hour","piece","moon","tight","near","scar","ride","someday","plan","way","meet","send","slip","cross","goin","comin","land","scream","straight","happen","end","brother","knee","water","drive","stone","drop","trust","breath","story","bear","felt","rise","count","stick","bone","warm","river","lady","anymore","lookin","wear","floor","trouble","choose","shake","shame","kick","window","answer","move","outside","livin","shine","whoa","fast","clear","peace","city","earth","picture","control","million","hop","worry","fell","future","teach","pretty","feelin","doubt","pride","different","sorrow","gettin","tryin","check","pick","power","gold","flow","mama","thousand","read","talkin","skin","tree","slow","thrill","bleed","sign","mouth","push","longer","lock","repeat","swear","rule","minute","sick","second","everyday","finally","evil","great","beautiful","hole","realize","mother","babe","pretend","shin","radio","mistake","brain","doin","lonesome","finger","forgive","whisper","return","slowly","sight","thought","guitar","fill","show","sorry","train","knock","sell","cloud","point","space","heartache","chain","band","cover","stranger","darkness","ghost","half","mountain","hair","flame","wide","desire","taste","bless","cool","treat","simple","ask","nice","winter","one","lovin","happiness","perfect","wild","remain","letter","sense","phone","raise","loud","bind","regret","survive","spring","daddy","rhyme","safe","deal","swing","crowd","snow","race","drown","number","ball","poor","bright","secret","awake","somethin","silence","wake","crawl","lovely","pay","human","kid","sky","closer","strange","magic","shut","bury","boy","runnin","darlin","fly","mirror","shadow","okay","question","track","shout","small","thinkin","heal","grave","school","shoe","fate","dirty","misery","book","single","funny","mess","christmas","wing","wonderful","paint","luck","tryna","father","ocean","weak","walkin","wave","shall","satisfy","shoulder","romance","travel","prayer","tongue","jump","style","clean","color","woah","state","sit","tender","remind","soft","ahead","brand","dust","lay","problem","tune","glass","escape","drift","fuckin","record","welcome","corner","country","wine","instead","fake","hello","paper","sand","green","block","crack","club","haunt","twist","let","suffer","dollar","prove","lifetime","warn","pack","crime","clock","freedom","draw","queen","climb","bend","spin","string","sail","scene","disappear","surprise","precious","ease","soldier","heavy","insane","even","makin","fact","sweetheart","cryin","oooh","spirit","family","sayin","outta","vision","sister","fail","decide","bust","news","sigh","golden","edge","youth","waitin","dress","guide","fair","bridge","paradise","strength","drag","bird","angel","figure","bite","wife","tie","freeze","heat","flower","teardrop","bottle","crash","wander","weep","silver","playin","rainbow","wrap","battle","quit","stronger","fee","holy","weather","short","loose","sink","strike","clothe","lesson","force","suppose","surely","grab","dont","rose","find","week","beg","pour","midnight","sittin","lift","choice","bell","seek","foolish","fallin","cash","higher","early","season","imagine","folk","charm","rock","proud","middle","vain","silent","chill","recall","trap","spit","suck","spot","surrender","beneath","loneliness","spell","path","stare","beauty","instrumental","slave","spread","harder","flesh","neck","view","drug","special","difference","pocket","bang","buy","enemy","cheat","note","ache","kinda","demon","certain","knife","bullet","lean","drum","twice","cast","hurry","motherfucker","suddenly","master","dare","ship","praise","sweat","hero","gun","quiet","smell","explain","tide","anybody","danger","breeze","echo","curse","tall","rhythm","circle","probably","wheel","quick","machine","crown","linger","weary","comfort","laughter","reality","babylon","judge","gim","embrace","ohoh","history","nation","rush","melt","tough","wise","destiny","garden","surround","worse","table","wicked","fever","couple","seven","hook","pussy","fame","glow","hood","weed","mend","name","message","mystery","momma","root","highway","pleasure","south","bitter","later","field","favorite","carry","diamond","emotion","rid","bomb","steel","cigarette","card","dope","chase","plain","fresh","takin","release","bother","business","final","hoe","goodnight","slide","moonlight","distance","dirt","self","chest","respect","cost","weight","mornin","freak","load","trick","deserve","swallow","destroy","sadness","gather","appear","hungry","sugar","join","asleep","worst","bore","softly","california","pressure","grey","divine","shed","movin","complete","wash","endless","offer","seed","bass","ash","gift","shape","test","nature","murder","fault","rockin","creep","pill","struggle","leavin","double","clown","telephone","amaze","grand","pound","hit","store","expect","valley","crew","texas","speed","notice","singin","type","boat","reveal","affair","passion","beer","steady","older","confuse","refuse","dancin","set","claim","brave","american","wire","admit","swim","gain","shatter","toss","crush","serve","act","hill","cowboy","bos","fantasy","beast","treasure","dumb","noise","month","form","flash","movie","miracle","ear","teeth","settle","america","create","mood","hat","float","erase","cling","stuff","daughter","wreck","disguise","kingdom","calm","needle","joke","wed","doctor","desert","stage","course","givin","rough","drinkin","cruel","underneath","east","greatest","vein","victim","bar","trade","tellin","lately","sin","shore","stupid","guilty","mighty","attack","compare","groove","pure","shade","put","whip","undo","workin","rapper","chair","bank","cut","part","nose","deeper","action","busy","solo","thunder","horse","sweeter","roam","hammer","ghetto","company","heartbeat","church","date","saturday","lick","callin","press","handle","chick","situation","wipe","island","faster","unknown","gang","food","flip","coffee","choke","exactly","boom","friday","seat","dime","marry","poison","nail","ought","direction","pillow","hardly","liar","wound","stain","scratch","car","deceive","smart","stumble","naked","monkey","dreamin","switch","funky","animal","bigger","sunday","beach","loser","march","forward","thee","yellow","coast","stack","daylight","prison","spark","anger","trail","pity","arrive","motion","hangin","excuse","trigger","borrow","autumn","whistle","cure","shelter","heartbreak"]
    new_lyrics = lyrics.split()
    if len(new_lyrics) > 3:
        bad_word= []
        for i in range(3):
            bad_word.append(new_lyrics.pop(random.randint(0,len(new_lyrics)-1)))
        for i in range(3):
            new_lyrics.append(words_to_try[random.randint(0,len(words_to_try)-1)])
    else:
        bad_word = None
        for i in range(3):
            new_lyrics.append(words_to_try[random.randint(0,len(words_to_try)-1)])
    
    new_lyrics = " ".join(new_lyrics)
    
    data = {"lyrics": new_lyrics, "genre": genre, "artist_name": artist_name, "track_name": track_name, "tempo": tempo, "cle": key , "explicit" : explicit, "duration" : duration}
    X = preprocess_data(data)
    
    new_score = predict_popularity(X)
    if new_score > score:
        if bad_word:
            improvement['lyrics'] = ("Change " + bad_word + " to " + new_lyrics, new_score)
        improvement['lyrics'] = ("add" + new_lyrics, new_score)
    else:
        if bad_word:
            good['lyrics'] = ("Change " + bad_word + " to " + new_lyrics, new_score)
        good['lyrics'] = ("add" + new_lyrics, new_score)
    
    return improvement, good
    
    
    

        
    
        
    
    
    
    