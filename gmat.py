import constants
import os
from collections import defaultdict
import random

# Pull student trajectories from GMAT data
# Currently they are promiscuously combined from inside_class and outside_class
def get_gmat_data():
    with open(os.path.join(constants.ROOT_PATH, constants.GMAT_COMBINED_DATA_PATH)) as f:
        lines = f.readlines()
        keys = lines[0]
        lines = lines[1:]

    data = defaultdict(list)
    with open(os.path.join(constants.ROOT_PATH, constants.GMAT_TOPICS_PATH)) as f:
        linestopics = f.readlines()
        linestopics = linestopics[1:]
        
    questionlist = []
    topicvecslist = []
    topicslist = ["simplicity", "logic", "modifier", 'subject predicate consistency', "verb tense", "accurate words", "parallel structure", "sentence structure", "comparative structure", "pronouns"]
    difficulties = []
    topicsveclistfinal = []
    
    for l in linestopics:
        topicvec = [0] * 10
        try:
            separated = l.split(',')
            question_id = int(separated[0])
            diff = separated[1]
            topics = separated[len(separated) - 1]      
        except Exception as e:
            print(l)
        listtopics = topics.split('#####')
        for category in listtopics:
            categorytopic = category.rstrip()
            if categorytopic in topicslist:
                ind = (topicslist).index(categorytopic)
                topicvec[ind] = 1
        if topicvec not in topicvecslist:
            topicvecslist.append(topicvec)
        questionlist.append(question_id) 
        difficulties.append(diff)
        topicsveclistfinal.append(topicvec)
    counter = 0
    # Assume data is sorted first by student ID, then by timestamp!
    # This will fail if that is not the case, because order will be all confused
    for l in lines:
        # Ignore fields which are unused (for now, at least)
        # Moreover, the last field - which is supposed to be the answer ID -
        # is sometimes a weird string. We have to explicitly throw those out
        # for now
        try:
            user_id, _, question_id, start_time, correct, _, _ = l.split(',')[:7]
        except Exception as e:
            print(l)

        user_id, question_id, start_time, correct = \
            int(user_id), int(question_id), int(start_time), int(correct)
                  
        topic_id_new = 0
        if question_id not in questionlist:
            continue
        else: 
            number = questionlist.index(question_id)
            difficulty = difficulties[number]
            topic_id = topicsveclistfinal[number] 
            
            base = 1
            for number in topic_id:
                topic_id_new += base * number
                base *= 2
                                       
            #if difficulty == "easy":
            #    topic_id[10] = 1
            #elif difficulty == "normal":
            #    topic_id[11] = 1
            #else:
            #    topic_id[12] = 1
            
        #print(user_id, question_id, duration, start_time, correct)
        line_dict = {
            'user_id' : user_id,
            'question_id' : question_id,
            'topic_id' : topic_id_new,
            'start_time' : start_time,
            'correct' : correct
        }
        if user_id in data and len(data[user_id]) > 0:
            assert(start_time >= data[user_id][-1]['start_time'])
        #if line_dict['topic_id'] != 0 and question_id % 10 == 0:
        data[user_id].append(line_dict)
        if (counter % 1000 == 0):
            print(line_dict)
        counter = counter + 1
    #print(data)
    return data
