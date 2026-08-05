import React, { createContext, useContext } from 'react';


export const NotebookHeaderContext = createContext({
  notebookHeader: null,
  setNotebookHeader: () => {},
  notebookActions: null,
  setNotebookActions: () => {},
  // Вкладки разделов держим отдельно от остальной шапки: имя и даты приходят
  // ответом сервера, а разделы известны сразу, и пропадать на время загрузки
  // блокнота навигация не должна.
  notebookTabs: null,
  setNotebookTabs: () => {},
});


export const useNotebookHeader = () => useContext(NotebookHeaderContext);
