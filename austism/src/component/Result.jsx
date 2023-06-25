import React from 'react';
import { Box, Typography, Button } from '@mui/material';

const Results = ({ result }) => {
    let declaration, color;

    if (result < 30) {
        declaration = "Based on the provided data, there is a low likelihood of ASD.";
        color = "green";
    } else if (result >= 30 && result < 50) {
        declaration = "Based on the provided data, there is a moderate likelihood of ASD.";
        color = "orange";
    } else {
        declaration = "Based on the provided data, there is a high likelihood of ASD.";
        color = "red";
    }

    return (
        <Box 
            display="flex"
            flexDirection="column"
            justifyContent="center"
            alignItems="center"
            height="80%"
            bgcolor="grey"
            borderRadius="10px"
            p={4}
        >
            <Typography variant="h3" mb={2}>Results</Typography>
            <Typography variant="h1" color={color} mb={4}>{result}</Typography>
            <Typography variant="h5" textAlign="center" mb={4}>{declaration}</Typography>
            <Button variant="contained" color="primary" onClick={() => window.location.reload()}>Start Over</Button>
        </Box>
    );
}

export default Results;
