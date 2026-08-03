import React from 'react';
import { AlertTriangle, Home, RefreshCw } from 'lucide-react';

import { Button } from './ui/Button';
import { useLocale } from '../i18n';

// Ловушка ошибок рендера обязана быть классовой: хуковых аналогов
// getDerivedStateFromError/componentDidCatch в React нет. Перевод берём из
// обёртки ниже — useLocale внутри класса не вызвать.
class ErrorBoundaryView extends React.Component {
    constructor(props) {
        super(props);
        this.state = { hasError: false };
    }

    static getDerivedStateFromError() {
        return { hasError: true };
    }

    componentDidCatch(error, errorInfo) {
        // Стек компонентов теряется вместе с деревом, поэтому пишем его сразу:
        // без него по одному сообщению ошибки место падения не найти.
        console.error('Unhandled render error', error, errorInfo?.componentStack);
    }

    handleReload = () => {
        window.location.reload();
    };

    handleGoHome = () => {
        window.location.assign('/');
    };

    render() {
        const { t, children } = this.props;

        if (!this.state.hasError) return children;

        return (
            <div className="flex min-h-screen items-center justify-center bg-slate-50 px-4 py-10">
                <div className="w-full max-w-md rounded-2xl border border-slate-200 bg-white p-6 text-center shadow-sm">
                    <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-xl bg-red-50 text-red-600">
                        <AlertTriangle className="h-6 w-6" />
                    </div>
                    <h1 className="text-lg font-semibold text-slate-900">{t('errorBoundary.title')}</h1>
                    <p className="mt-2 text-sm text-slate-500">{t('errorBoundary.description')}</p>
                    <div className="mt-6 flex flex-col gap-3 sm:flex-row sm:justify-center">
                        <Button type="button" onClick={this.handleReload}>
                            <RefreshCw className="h-4 w-4" />
                            {t('errorBoundary.reload')}
                        </Button>
                        <Button type="button" variant="outline" onClick={this.handleGoHome}>
                            <Home className="h-4 w-4" />
                            {t('errorBoundary.home')}
                        </Button>
                    </div>
                </div>
            </div>
        );
    }
}

const ErrorBoundary = ({ children }) => {
    const { t } = useLocale();

    return <ErrorBoundaryView t={t}>{children}</ErrorBoundaryView>;
};

export default ErrorBoundary;
