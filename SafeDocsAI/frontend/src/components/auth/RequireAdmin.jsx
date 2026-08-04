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
 *
 * Текст объяснения задаётся ключом перевода: «раздел только для админов» —
 * правда не про каждый закрытый раздел, а неверное объяснение отправляет
 * пользователя не к тому человеку.
 */
export const AccessDenied = ({ descriptionKey = 'access.deniedDescription' }) => {
    const navigate = useNavigate();
    const { t } = useLocale();

    return (
        <div className="mx-auto max-w-xl rounded-2xl border border-slate-200 bg-white p-8 text-center shadow-sm">
            <div className="mx-auto mb-4 flex h-12 w-12 items-center justify-center rounded-full bg-amber-100 text-amber-600">
                <ShieldAlert className="h-6 w-6" />
            </div>
            <h2 className="text-xl font-extrabold text-[#1f3a60]">{t('access.deniedTitle')}</h2>
            <p className="mt-2 text-sm text-slate-500">{t(descriptionKey)}</p>
            <Button type="button" variant="outline" className="mt-5" onClick={() => navigate('/chat')}>
                {t('access.backToChat')}
            </Button>
        </div>
    );
};

/**
 * Общая механика гарда: пока роль выясняется — «проверяем», дальше либо раздел,
 * либо честная заглушка.
 *
 * Вынесена из RequireAdmin, когда понадобился второй набор ролей. Отдельный
 * гард-близнец рядом был бы копией с одним изменённым сравнением, и разошлись
 * бы они на первой же правке — например, на состоянии «роль ещё выясняется»,
 * без которого заглушка успевает мигнуть перед законным владельцем прав.
 */
const RoleGate = ({ allowed, deniedDescriptionKey }) => {
    const { t } = useLocale();

    if (allowed === null) {
        return (
            <div className="rounded-2xl border border-slate-200 bg-white p-8 text-center text-slate-500 shadow-sm">
                {t('access.checking')}
            </div>
        );
    }

    return allowed ? <Outlet /> : <AccessDenied descriptionKey={deniedDescriptionKey} />;
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

    return <RoleGate allowed={isResolving ? null : isAdmin} deniedDescriptionKey="access.deniedDescription" />;
};

/**
 * Маршруты, которые ведёт контент-менеджер: сейчас это презентации блокнота.
 *
 * Почему не параметр у RequireAdmin: гард с именем «требуется администратор»,
 * пускающий контент-менеджера, читается как ошибка ровно там, где его увидят —
 * в таблице маршрутов. Почему не отдельный файл: механика у обоих гардов одна
 * (RoleGate выше), и живут они в одном модуле, чтобы состояние «роль ещё
 * выясняется» правилось в одном месте для обоих.
 *
 * Набор ролей объявлен не здесь, а в useSessionRole (canManageContent): тот же
 * вопрос «может ли этот пользователь управлять содержимым» задаёт и вкладка
 * блокнота, которой гард не нужен — ей нужно только знать, рисовать ли ссылку.
 */
export const RequireContentAccess = () => {
    const { canManageContent, isResolving } = useSessionRole();

    return (
        <RoleGate
            allowed={isResolving ? null : canManageContent}
            deniedDescriptionKey="access.deniedContentDescription"
        />
    );
};

export default RequireAdmin;
