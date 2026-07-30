import React, { createContext, useContext } from 'react';


export const NotebookHeaderContext = createContext({
  notebookHeader: null,
  setNotebookHeader: () => {},
  notebookActions: null,
  setNotebookActions: () => {},
});


export const useNotebookHeader = () => useContext(NotebookHeaderContext);
