import { useState } from "react";
import Question from "./quenstion";
import Question2 from "./Question2";
import Results from "./Result";
import { Box, Typography, Button } from "@mui/material";

const MyForm = () => {
  const questions = [
    {
      title: "S/he often notices small sounds when others do not",
      value: 0,
    },
    {
      title:
        "S/he usually concentrates more on the whole picture, rather than the small details",
      value: 1,
    },
    {
      title:
        "In a social group, s/he can easily keep track of several different people`s conversations",
      value: 1,
    },
    {
      title:
        "S/he finds it easy to go back and forth between different activities",
      value: 1,
    },
    {
      title:
        "S/he doesn`t know how to keep a conversation going with his/her peers",
      value: 0,
    },
    {
      title: "S/he is good at social chit-chat",
      value: 1,
    },
    {
      title:
        "When s/he is read a story, s/he finds it difficult to work out the character`s intentions or feelings",
      value: 0,
    },
    {
      title:
        "When s/he was in preschool, s/he used to enjoy playing games involving pretending with other children",
      value: 1,
    },
    {
      title:
        "S/he finds it easy to work out what someone is thinking or feeling just by looking at their face",
      value: 1,
    },
    {
      title: "S/he finds it hard to make new friends",
      value: 0,
    },
  ];
  const [form, setForm] = useState({
    A1_Score: 0,
    A2_Score: 0,
    A3_Score: 0,
    A4_Score: 0,
    A5_Score: 0,
    A6_Score: 0,
    A7_Score: 0,
    A8_Score: 0,
    A9_Score: 0,
    A10_Score: 0,
    age: 0,
    gender: "",
    ethnicity: "",
    autism: "",
    result: 0,
  });

  const [finalResult, setFinalResult] = useState(null);

  const [currentQuestion, setCurrentQuestion] = useState(0);

  const onNext = (questionIndex, value) => {
    const result = (form.A1_Score + form.A2_Score + form.A3_Score + form.A4_Score + form.A5_Score + form.A6_Score + form.A7_Score + form.A8_Score + form.A9_Score + form.A10_Score ) *2;
    const questionKey = `A${questionIndex + 1}_Score`;
    setForm({
      ...form,
      [questionKey]: value,
        result: result,
    });
    setCurrentQuestion(questionIndex + 1);
  };

  const handleChange = (event) => {
    
    setForm({
      ...form,
      [event.target.name]: event.target.value,
    });
  };

  const handleSubmit = (event) => {
    

    event.preventDefault();
    console.log(form);
    fetch("http://localhost:5000/submit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(form),
    })
      .then((response) => response.json())
      .then((data)=>{ console.log(data);
        setFinalResult(data.prediction);
  }) 
      .catch((error) => {
        console.error("Error:", error);
      });
  };



  if (finalResult) {
    return (
      <div>
        <Results result={finalResult} />
      </div>
    );
  } else {
    return currentQuestion < questions.length ? (
      <Question
        questionText={questions[currentQuestion].title}
        questionIndex={currentQuestion}
        onNext={onNext}
        value={questions[currentQuestion].value}
      />
    ) : (
        <Box 
            display="flex"
            flexDirection="column"
            justifyContent="center"
            alignItems="center"
            height='80%'
            width='30%'
            bgcolor="grey"
            borderRadius="10px"
            p={4}
        >
      <form onSubmit={handleSubmit}>
        <Question2 field="age" handleChange={handleChange} value={form.age} />
        <Question2
          field="gender"
          handleChange={handleChange}
          value={form.gender}
        />
        <Question2
          field="autism"
          handleChange={handleChange}
          value={form.autism}
        />
        <Question2
          field="ethnicity"
          handleChange={handleChange}
          value={form.ethnicity}
        />
        <button type="submit">Submit</button>
      </form>
      </Box>
    );
  }
};

export default MyForm;
