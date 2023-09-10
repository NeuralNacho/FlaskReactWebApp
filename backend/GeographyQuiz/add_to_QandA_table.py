from .database_models import QuestionAndAnswer
from sqlalchemy import create_engine, MetaData
from sqlalchemy.orm import sessionmaker
import os

def add_to_QandA_table(question, answer, answer_info, question_category):
    db_filename = os.path.join('instance', 'geography_quiz_database.db')
    db_url = f'sqlite:///{db_filename}' 

    # Create a database engine
    engine = create_engine(db_url)
    QuestionAndAnswer.metadata.create_all(engine)
    Session = sessionmaker(bind = engine)
    session = Session()

    existing_question = session.query(QuestionAndAnswer).filter_by(question = question).first()

    if existing_question is None:
        # If the question is not found, add it to the database
        question_and_answer = QuestionAndAnswer(
            question = question,
            question_category = question_category,
            answer = answer, 
            answer_info = answer_info
        )

        session.add(question_and_answer)
        session.commit()
    else:
        print(f"Question '{question}' already exists in the database. Skipping insertion.")
    
    session.close()

def delete_QandA_table():
    db_url = 'sqlite:///geography_quiz_database.db'

    # Create a database engine
    engine = create_engine(db_url)

    # Create a metadata object
    metadata = MetaData()

    # Bind the engine to the metadata
    metadata.reflect(bind = engine)

    # Get the table object based on your model
    table = metadata.tables.get('question_and_answer')  # Replace with your table name

    if table is not None:
        # Drop the table
        table.drop(engine)

if __name__ == '__main__':
    questions = ['What is the second longest river in South America?', 
                'Which language is this: गिव में सम हिंदी', 
                'What is the population density of India /km²? (Get within 10)']
    answers = ['Parana', 'Hindi', '435']
    answer_info = ['The Paraná River is a river in south-central South America, \
                    running through Brazil, Paraguay, and Argentina for some 4,880 kilometres.',
                    'Modern Standard Hindi commonly referred to as Hindi, is an \
                    Indo-Aryan language spoken chiefly in North India.',
                    'The population density of India is 435. This ranks 19th out \
                    of all countries in the world.']
    question_categories = ['River name', 'Guess language', 'Population density']

    for i in range(len(questions)):
        add_to_QandA_table(questions[i], answers[i], answer_info[i], question_categories[i])