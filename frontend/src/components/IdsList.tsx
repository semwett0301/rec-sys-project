import React from "react";
import {Box, CircularProgress, Divider, List, ListItem, ListItemText, Paper, Typography} from "@mui/material";

interface NamesListProps {
    names: string[];
    loading: boolean;
}

export const IdsList: React.FC<NamesListProps> = ({names, loading}) => {
    if (loading) {
        return (
            <Box display="flex" justifyContent="center" mt={4}>
                <CircularProgress/>
            </Box>
        );
    }

    return (
        <Paper elevation={2} sx={{mt: 4, p: 2}}>
            <Typography variant="h6" gutterBottom>The list of item ids</Typography>
            {names.length === 0 ? (
                <Typography color="textSecondary">No names to display. Add some names!</Typography>
            ) : (
                <List>
                    {names.map((name, index) => (
                        <React.Fragment key={name}>
                            <ListItem>
                                <ListItemText
                                    primary={name}
                                />
                            </ListItem>
                            {index < names.length - 1 && <Divider/>}
                        </React.Fragment>
                    ))}
                </List>
            )}
        </Paper>
    );
};
