from models.basemodel import BaseModel
import numpy as np
import tensorflow as tf

#work in progress
class DKTModel(BaseModel):
    def __init__(self):
        self.DROPOUT = 0.5
        self.MAX_EPOCHS = 15
        self.BATCH_SIZE = 32
        self.MAX_LENGTH = 350
        self.LR = 0.001
        self.HIDDEN_SIZE = 200
        self.NUM_QUESTIONS = 8543
        self.EMBEDDING_SIZE = 25
        self.setup()
           
    def addplaceholders(self):
        self.questions_placeholder = tf.placeholder(tf.int32,(None, self.MAX_LENGTH))
        self.answers_placeholder = tf.placeholder(tf.int32, (None, self.MAX_LENGTH))
        self.lengths_placeholder = tf.placeholder(tf.int32, (None))
        self.mask_placeholder = tf.placeholder(tf.bool, (None, self.MAX_LENGTH))
        self.dropout_placeholder = tf.placeholder_with_default(self.DROPOUT, ())
        self.lr_placeholder = tf.placeholder_with_default(self.LR, ())
        
    def data_pipeline(self):
        xav_init = tf.contrib.layers.xavier_initializer()
        d = 1.0 - self.dropout_placeholder      
        embeddings = tf.Variable(tf.random_uniform([self.NUM_QUESTIONS * 2, self.EMBEDDING_SIZE], -1.0, 1.0))
        init_state1 = tf.Variable(tf.random_uniform([self.BATCH_SIZE, self.HIDDEN_SIZE], -1.0, 1.0))
        init_state2 = tf.Variable(tf.random_uniform([self.BATCH_SIZE, self.HIDDEN_SIZE], -1.0, 1.0))
        batch_size = tf.shape(self.questions_placeholder)[0]
        answers_no_start = self.questions_placeholder + self.NUM_QUESTIONS * self.answers_placeholder
        all_answers = tf.concat((tf.zeros((batch_size, 1), dtype=tf.int32), answers_no_start), axis=1)
        answer_seqs = tf.nn.embedding_lookup(embeddings, all_answers)
        self.seqs = answer_seqs
        with tf.variable_scope('lstm1'):
            cell1 = tf.contrib.rnn.LSTMCell(self.HIDDEN_SIZE, reuse = None)
            cell1 = tf.contrib.rnn.DropoutWrapper(cell1, output_keep_prob=d)
            outputs1, hidden_states = tf.nn.dynamic_rnn(cell=cell1, inputs=self.seqs, sequence_length=self.lengths_placeholder + 1, dtype=tf.float32,swap_memory=True)
        with tf.variable_scope('lstm2'):
            cell2 = tf.contrib.rnn.LSTMCell(self.HIDDEN_SIZE, reuse = None)
            cell2 = tf.contrib.rnn.DropoutWrapper(cell2, output_keep_prob=d)
            outputs, hidden_states = tf.nn.dynamic_rnn(cell=cell2, inputs=outputs1, sequence_length=self.lengths_placeholder + 1, dtype=tf.float32,swap_memory=True)        
        w = tf.get_variable("W", (self.HIDDEN_SIZE, self.NUM_QUESTIONS), tf.float32, xav_init)
        b = tf.get_variable("b", (self.NUM_QUESTIONS,), tf.float32, tf.constant_initializer(0.0))
        outputs_flat = tf.reshape(outputs, [-1, self.HIDDEN_SIZE])
        inner = tf.matmul(outputs_flat, w) + b
        self.all_probs = tf.sigmoid(tf.reshape(inner, [-1, self.MAX_LENGTH + 1, self.NUM_QUESTIONS]))
        self.post_probs = self.all_probs[:, self.lengths_placeholder[0]] # assumes all seq_lens are equal
        self.probs = tf.slice(self.all_probs, [0,0,0], [-1, self.MAX_LENGTH, -1])
        self.v_hats = tf.reduce_sum(self.probs, 2)
        question_indicators = tf.one_hot(self.questions_placeholder, self.NUM_QUESTIONS)
        self.question_probs = tf.reduce_sum(self.probs * question_indicators, 2)
        return self.question_probs
   
    def eval_probs(self):
        guesses = tf.to_int32(tf.round(self.question_probs))
        corrects = tf.to_int32(tf.equal(guesses, self.answers_placeholder))
        masked_corrects = tf.boolean_mask(corrects, self.mask_placeholder)
        num_correct = tf.reduce_sum(masked_corrects)
        #num_correct = tf.reduce_sum(corrects)
        num_total = tf.reduce_sum(tf.to_int32(self.mask_placeholder))        
        auc_score, auc_op = tf.metrics.auc(self.answers_placeholder, self.question_probs)
        return num_correct, num_total, auc_score, auc_op
               
    def find_loss(self, probs, labels):
        losses = probs - tf.to_float(labels)
        mask_losses = tf.boolean_mask(losses, self.mask_placeholder)
        total_loss = tf.reduce_sum(mask_losses**2)
        #total_loss = tf.reduce_sum(losses)
        return total_loss
               
    def setup(self):
        self.addplaceholders()
        self.question_probs = self.data_pipeline()
        self.num_correct, self.num_total, self.auc_score, self.auc_op = self.eval_probs()
        self.loss = self.find_loss(self.question_probs, self.answers_placeholder)
        self.train_op = tf.train.AdamOptimizer(self.lr_placeholder).minimize(self.loss) #compute gradients
        #gradient clip by global norm ...10
        #apply gradients
        self.saver = tf.train.Saver()
        
    def train_on_batch(self, session, lengths, masks, answers, questions):
        feed_dict = {self.lengths_placeholder: lengths,
                     self.mask_placeholder: masks,
                     self.answers_placeholder: answers,
                     self.questions_placeholder: questions}
        _, loss = session.run([self.train_op, self.loss], feed_dict = feed_dict)
        return loss
        
    def test_on_batch(self, session, lengths, masks, answers, questions):
        feed_dict = {self.lengths_placeholder: lengths,
                     self.mask_placeholder: masks,
                     self.answers_placeholder: answers,
                     self.questions_placeholder: questions}
        num_correct, num_total, auc, v_hats = session.run([self.num_correct, self.num_total, self.auc_op, self.v_hats], feed_dict = feed_dict)
        return num_correct, num_total, auc, v_hats
    
    # takes in the lists of answer sequences and corresponding question sequences, pads t
    def processdata(self, answers, questions, num_questions):
        assert(len(answers) == len(questions))
        n = len(answers)
        questions_padded = []
        answers_padded = []
        masks = []
        lengths = []     
        for i in range(n):
            assert(len(answers[i]) == len(questions[i]))
            lencur = len(answers[i])
            if lencur > self.MAX_LENGTH:
                questions[i] = questions[i][:self.MAX_LENGTH]
                answers[i] = answers[i][:self.MAX_LENGTH]
                lencur = self.MAX_LENGTH
            lengths.append(lencur)
            #pads remaining question spots with 0s, but the mask indicates the padded spots
            padding = [0] * (self.MAX_LENGTH - lencur)
            #paddingquestions = [[0] * 10] * (self.MAX_LENGTH - lencur)
            answers_padded.append(answers[i] + padding)
            questions_padded.append(questions[i] + padding)
            masks.append([1] * lencur + padding)            
        answers = answers_padded
        questions = questions_padded
        return lengths, masks, answers, questions
            
    def batchify(self, zipped_data):
        num_batches =  int(np.ceil(len(zipped_data) / self.BATCH_SIZE))
        num_fields = len(zipped_data[0])
        batches = []
        for i in range(num_batches):
            batch_data = zipped_data[i*self.BATCH_SIZE:(i+1)*self.BATCH_SIZE]
            batch = [[] for i in range(len(batch_data[0]))]
            for b in batch_data:
                for j in range(num_fields):
                    batch[j].append(b[j])
                batches.append(batch)
        return batches
    
    def train(self, train_data, eval_data): 
        with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            self.train_model(self, session, train_data, eval_data)
        
    def train_model(self, model, session, train_data, eval_data):
        processed = self.processdata(train_data['responses'], train_data['question_ids_modified'], self.NUM_QUESTIONS)
        lens, masks, answers, questions = processed
        assert(len(lens) == len(masks) == len(answers) == len(questions))
        zipped_data = list(zip(*processed))
        for epoch in range(self.MAX_EPOCHS):
            np.random.shuffle(zipped_data)
            train_batches = self.batchify(zipped_data)
            for train_batch in train_batches:
                loss = model.train_on_batch(session, *train_batch) 
                print(loss)
            self.eval_model(model, session, eval_data)

    def eval(self, eval_data):
         with tf.Session() as session:
            session.run(tf.global_variables_initializer())
            session.run(tf.local_variables_initializer())
            #self.eval_model(session, eval_data)
    
    def eval_model(self, model, session, eval_data):
        processed = self.processdata(eval_data['responses'], eval_data['question_ids_modified'], self.NUM_QUESTIONS)
        lens, masks, answers, questions = processed
        assert(len(lens) == len(masks) == len(answers) == len(questions))
        zipped_data = list(zip(*processed))
        np.random.shuffle(zipped_data)
        test_batches = self.batchify(zipped_data)
        correct = 0.
        attempted = 0.
        total_auc = 0.
        for test_batch in test_batches:
            num_correct, num_total, auc, v_hats = model.test_on_batch(session, *test_batch)
            total_auc += num_total * auc
            correct += num_correct
            attempted += num_total
            print(correct)
            print(attempted)       
        print("{} test examples right out of {}, which is {} percent. Mean AUC {}\n".format(
            correct, attempted, 100.0*correct/attempted, 1.0*total_auc/attempted))
        return correct/attempted
            
