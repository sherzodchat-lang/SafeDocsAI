import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate, Outlet, useParams } from 'react-router-dom';
import ErrorBoundary from './components/ErrorBoundary';
import Layout from './components/layout/Layout';
import LoginPage from './pages/LoginPage';
import ChatPage from './pages/ChatPage';
import AskPage from './pages/AskPage';
import NotesPage from './pages/NotesPage';
import InsightsPage from './pages/InsightsPage';
import AdminSourcesPage from './pages/AdminSourcesPage';
import AdminLogsPage from './pages/AdminLogsPage';
import NotebooksPage from './pages/NotebooksPage';
import NotebookLayout from './pages/NotebookLayout';
import NotebookOverviewPage from './pages/NotebookOverviewPage';
import NotebookPresentationsPage from './pages/NotebookPresentationsPage';
import SettingsPage from './pages/SettingsPage';
import TopicsPage from './pages/TopicsPage';

import RegisterPage from './pages/RegisterPage';
import RequireAdmin, { RequireContentAccess } from './components/auth/RequireAdmin';
import { hasActiveSession } from './services/api';

// Токены лежат в httpOnly-куках и из JS не читаются, поэтому признаком входа
// служит читаемая CSRF-кука: её бэкенд ставит и гасит вместе с ними.
const PrivateRoute = () => (hasActiveSession() ? <Outlet /> : <Navigate to="/login" replace />);

const NotebookChatRoute = () => {
  const { notebookId } = useParams();
  return <ChatPage notebookId={Number(notebookId)} />;
};

const NotebookSourcesRoute = () => {
  const { notebookId } = useParams();
  return <AdminSourcesPage notebookId={Number(notebookId)} />;
};

function App() {
  return (
    // Граница стоит над роутером: ошибка рендера любой страницы должна давать
    // понятный экран, а не пустой белый документ.
    <ErrorBoundary>
      <Router>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected Routes */}
          <Route element={<PrivateRoute />}>
            <Route path="/" element={<Layout />}>
              <Route index element={<Navigate to="/chat" replace />} />
              {/* Без явного notebookId глобальный чат берёт активный блокнот из
                  localStorage — тот же, что выбирается на странице «Блокноты». */}
              <Route path="chat" element={<ChatPage />} />
              <Route path="sources" element={<AdminSourcesPage notebookId={null} />} />
              {/* Темы — надстройка над теми же источниками: экран открыт всем,
                  правами ограничена только переразметка внутри него. */}
              <Route path="topics" element={<TopicsPage />} />
              <Route path="ask" element={<AskPage />} />
              <Route path="notes" element={<NotesPage />} />
              <Route path="insights" element={<InsightsPage />} />
              <Route path="notebooks" element={<NotebooksPage />} />
              <Route path="notebooks/:notebookId" element={<NotebookLayout />}>
                <Route index element={<NotebookOverviewPage />} />
                <Route path="chat" element={<NotebookChatRoute />} />
                <Route path="sources" element={<NotebookSourcesRoute />} />
                {/* Заметки блокнота живут в «Обзоре» — там они умеют правку,
                    архивацию и удаление. Старый адрес не оставляем мёртвым:
                    ссылка из закладок или письма ведёт туда, где работа с
                    заметками теперь и идёт, а не в пустой экран. */}
                <Route path="notes" element={<Navigate to=".." relative="path" replace />} />
                {/* Презентации бэкенд отдаёт контент-менеджеру и админу
                    (presentation.role_not_allowed остальным). Гард здесь
                    ничего не разрешает: он лишь заменяет форму, из которой всё
                    равно ничего не отправится, на объяснение — прямой заход по
                    адресу пользовательской ролью упирается именно в него. */}
                <Route element={<RequireContentAccess />}>
                  <Route path="presentations" element={<NotebookPresentationsPage />} />
                </Route>
              </Route>
              <Route path="admin/sources" element={<AdminSourcesPage />} />
              <Route path="admin/documents" element={<Navigate to="/admin/sources" replace />} />
              {/* Журнал и настройки бэкенд отдаёт только админу
                  (deps.get_current_active_superuser). Проверка здесь ничего не
                  разрешает — она лишь заменяет заведомо неработающую страницу
                  на честную заглушку. */}
              <Route element={<RequireAdmin />}>
                <Route path="admin/logs" element={<AdminLogsPage />} />
                <Route path="settings" element={<SettingsPage />} />
              </Route>
            </Route>
          </Route>

          {/* Fallback */}
          <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
      </Router>
    </ErrorBoundary>
  );
}

export default App;
