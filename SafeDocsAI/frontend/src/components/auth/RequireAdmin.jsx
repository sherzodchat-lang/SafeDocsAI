import React from 'react';
import { Outlet, useNavigate } from 'react-router-dom';
import { ShieldAlert } from 'lucide-react';
import { Button } from '../ui/Button';
import { useSessionRole } from '../../hooks/useSessionRole';
import { useLocale } from '../../i18n';

/**
 * Заглушка вместо админского раздела. Показывает ровно то, что есть: раздел
 * существует, но текущей роли он не положен, — вместо отрисованной формы, в
 * которой всё равно ничего не сохранится.
 */
export const AccessDenied = () => {
    const navigate = useNavigate();
    const { t } = useLocale();

    return (
        <div className="mx-auto max-w-xl rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-amber-100 text-amber-600">
                <ShieldAlert className="h-6 w-6" />
            </div>
            <h2 className="text-xl font-extrabold text-[#1f3a60]">{t('access.deniedTitle')}</h2>
            <p className="mt-2 text-sm text-slate-500">{t('access.deniedDescription')}</p>
            <Button type="button" variant="outline" className="mt-5" onClick={() => navigate('/chat')}>
                {t('access.backToChat')}
            </Button>
        </div>
    );
};

/**
 * Клиентская проверка роли для админских маршрутов.
 *
 * Границей безопасности остаётся сервер: он проверяет права на каждом запросе
 * и отвечает 403 независимо от того, что решил браузер. Здесь решается только
 * вопрос отрисовки — показывать раздел или заглушку.
 */
const RequireAdmin = () => {
    const { isAdmin, isResolving } = useSessionRole();
    const { t } = useLocale();

    if (isResolving) {
        return (
            <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center text-slate-500 shadow-sm">
                {t('access.checking')}
            </div>
        );
    }

    return isAdmin ? <Outlet /> : <AccessDenied />;
};

export default RequireAdmin;
