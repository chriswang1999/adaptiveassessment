import constants

class StudentData:
    def __init__(self, data, description):
        self.full_data = data
        self.train_data = {k:v for (k,v) in data.items() if k not in constants.TEST_USER_IDS}
        self.test_data =  {k:v for (k,v) in data.items() if k in constants.TEST_USER_IDS}
        print(len(self.train_data), len(self.test_data))
        self.description = description

    def training_trajectories(self, start_fraction = 0., end_fraction = 1.):
        corrects = []
        question_ids = []
        question_ids_modified = []
        dictionary = []
        for d in self.train_data.values():
            cur_corrects = []
            cur_question_ids = []
            cur_question_ids_modified = []
            for p in d:
                cur_corrects.append(p['correct'])
                cur_question_ids.append(p['question_id'])
                if p['question_id'] not in dictionary:
                    dictionary.append(p['question_id'])
                cur_question_ids_modified.append(dictionary.index(p['question_id']))
            corrects.append(cur_corrects)
            question_ids.append(cur_question_ids)
            question_ids_modified.append(cur_question_ids_modified)                                    
        start_index = int(start_fraction * len(self.train_data.values()))
        end_index =   int(end_fraction   * len(self.train_data.values()))
        return {
            'responses' : corrects[start_index : end_index],
            'question_ids' : question_ids[start_index : end_index],
            'question_ids_modified': question_ids_modified[start_index: end_index]
        }

