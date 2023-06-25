import { TextField, Select, MenuItem, FormControl, InputLabel } from "@mui/material";

const Question2 = ({ field, handleChange, value }) => {
  let input;

  switch (field) {
    case "age":
      input = (
        <TextField
          type="number"
          name="age"
          label="Age"
          variant="outlined"
          fullWidth
          onChange={handleChange}
          value={value}
        />
      );
      break;
    case "gender":
    case "autism":
      const options = field === "gender" ? [{"name":"Male","value":"m"},{"name": "Female", "value": "f"}] : [{"name":"Yes","value":"yes"},{"name": "No", "value": "no"}];
      input = (
        <FormControl fullWidth variant="outlined">
          <InputLabel>{field}</InputLabel>
          <Select name={field} value={value} onChange={handleChange} label={field==="autism"? "Autism Family History?" : "Gender"}>
            {options.map((option) => (
              <MenuItem key={option} value={option.value}>
                {option.name}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      );
      break;
    case "ethnicity":
      input = (
        <FormControl fullWidth variant="outlined">
          <InputLabel>Ethnicity</InputLabel>
          <Select
            name="ethnicity"
            value={value}
            onChange={handleChange}
            label="Ethnicity"
          >
            <MenuItem value="White-European">White-European</MenuItem>
            <MenuItem value="Latino">Latino</MenuItem>
            <MenuItem value="Others">Others</MenuItem>
            <MenuItem value="Black">Black</MenuItem>
            <MenuItem value="Asian">Asian</MenuItem>
            <MenuItem value="Middle Eastern">Middle Eastern</MenuItem>
          </Select>
        </FormControl>
      );
      break;
    default:
      input = null;
  }

  return (
    <div className="mb-4">
      <div className="mb-2 text-xl font-bold">{field==="autism"? "Autism Family History?" : field}</div>
      {input}
    </div>
  );
};

export default Question2;
