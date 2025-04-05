import React, {useState} from 'react';
import {
    Box,
    Button,
    CircularProgress,
    Container,
    FormControl,
    InputLabel,
    MenuItem,
    Paper,
    Select,
    SelectChangeEvent,
    TextField,
    Typography
} from '@mui/material';
import {IdsList} from "./components/IdsList.tsx";
import {Model} from "./types/Model.ts";
import {Dataset} from "./types/Dataset.ts";
import {ServerResponse} from "./types/ServerResponse.ts";

const models: Model[] = [
    {
        value: 'svd',
        label: 'SVD++'
    },
    {
        value: 'collab',
        label: 'Collaborative ItemKNN'
    },
    {
        value: 'content-user',
        label: 'Content-based UserKNN'
    },
    {
        value: 'content-item',
        label: 'Content-based ItemKNN'
    },
]

const datasets: Dataset[] = [
    {
        label: 'Yelp',
        value: 'yelp',
        allow: ['svd', 'content-user', 'content-item']
    },
    {
        label: 'MovieLens',
        value: 'movie',
        allow: ['svd', 'collab', 'content-user', 'content-item']
    },
    {
        label: 'Netflix',
        value: 'netflix',
        allow: ['collab']
    },
]

const BASE_URL = "http://127.0.0.1:8000/recommend";

const App: React.FC = () => {
    const [model, setModel] = useState<string>('');
    const [dataset, setDataset] = useState<string>('');
    const [userId, setUserId] = useState<string>('');
    const [n, setN] = useState<number>(0);
    const [loading, setLoading] = useState<boolean>(false);

    const [names, setNames] = useState<string[]>([])

    const workingDataset = datasets.filter((el) => !!el.allow.find((cand) => cand === model))

    const handleModelChange = (event: SelectChangeEvent) => {
        setModel(event.target.value);
        setDataset('')
    };

    const handleDatasetChange = (event: SelectChangeEvent) => {
        setDataset(event.target.value);
    };

    const handleUserIdChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setUserId(event.target.value);
    };

    const handleNChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const new_n = +event.target.value
        new_n > 0 && setN(+event.target.value);
    };

    const handleSubmit = async (event: React.FormEvent) => {
        event.preventDefault();
        if (!userId.trim()) return;
        setLoading(true);

        try {
            const response = await fetch(`${BASE_URL}/${model}/${dataset}?user_id=${userId}&top_n=${n}`);
            const data: ServerResponse = await response.json();

            setNames(data.recommendations)
        } finally {
            setLoading(false)
        }

    };

    return (
        <Container maxWidth="md" sx={{py: 4}}>
            <Typography variant="h4" component="h1" gutterBottom>
                Top-N tester
            </Typography>

            <Paper elevation={3} sx={{p: 3}}>
                <Box component="form" onSubmit={handleSubmit} noValidate>
                    <FormControl fullWidth margin="normal">
                        <TextField
                            label="User id"
                            value={userId}
                            onChange={handleUserIdChange}
                            variant="outlined"
                            required
                            disabled={loading}
                        />
                    </FormControl>
                    <FormControl fullWidth margin="normal">
                        <TextField
                            label="N"
                            value={n}
                            onChange={handleNChange}
                            variant="outlined"
                            required
                            type="number"
                            disabled={loading}
                        />
                    </FormControl>

                    <Box display="flex" gap={2} mt={2}>
                        <FormControl fullWidth margin="normal">
                            <InputLabel id="model-label">Model</InputLabel>
                            <Select
                                labelId="model-label"
                                value={model}
                                onChange={handleModelChange}
                                label="Model"
                                disabled={loading}
                                required
                                variant="outlined"
                            >
                                {
                                    models.map(({value, label}) => (
                                        <MenuItem value={value}>{label}</MenuItem>
                                    ))
                                }
                            </Select>
                        </FormControl>

                        <FormControl fullWidth margin="normal">
                            <InputLabel id="dataset-label">Dataset</InputLabel>
                            <Select
                                labelId="dataset-label"
                                value={dataset}
                                onChange={handleDatasetChange}
                                label="Dataset"
                                disabled={loading}
                                required
                                variant="outlined"
                            >
                                {
                                    workingDataset.map(({value, label}) => (
                                        <MenuItem value={value}>{label}</MenuItem>
                                    ))
                                }
                            </Select>
                        </FormControl>
                    </Box>

                    <Box mt={3} display="flex" justifyContent="flex-end">
                        <Button
                            type="submit"
                            variant="contained"
                            color="primary"
                            disabled={loading || !userId.trim() || !model || !dataset}
                            startIcon={loading && <CircularProgress size={20} color="inherit"/>}
                        >
                            {loading ? 'Submitting...' : 'Submit'}
                        </Button>
                    </Box>
                </Box>
            </Paper>

            <IdsList names={names} loading={loading}/>
        </Container>
    );
};

export default App;