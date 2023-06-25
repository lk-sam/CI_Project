import { useState } from 'react';
import { Button } from '@mui/material';

const Question = ({ questionText, questionIndex, onNext, value }) => {
    const [selected, setSelected] = useState(null);

    const onSelect = (score) => {
        setSelected(score);
        onNext(questionIndex, score);
    };

    return (
        <>
            <h1 className='mb-8 text-4xl text-center font-bold'>
                 Question {questionIndex + 1} 
            </h1>

            <p className='mb-8 text-center text-lg font-light'>
                {questionText}
            </p>

            <Button variant='outlined' className='mb-2' color='primary' fullWidth onClick={() => onSelect(value)}>
                Definitely Agree
            </Button>

            <Button variant='outlined' className='mb-2' color='primary' fullWidth onClick={() => onSelect(1 - value)}>
                Slightly Agree
            </Button>

            <Button variant='outlined' className='mb-2' color='error' fullWidth onClick={() => onSelect(1 - value)}>
                Slightly Disagree
            </Button>

            <Button variant='outlined' className='mb-2' color='error' fullWidth onClick={() => onSelect(value)}>
                Definitely Disagree
            </Button>
        </>
    );
};

export default Question;
