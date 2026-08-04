import { useEffect, useState } from 'react';
import { ensureSessionRole, getSessionRole, hasActiveSession, subscribeSessionRole } from '../services/api';

/**
 * Роль текущей сессии для интерфейса.
 *
 * Права по-прежнему проверяет сервер: роль здесь нужна только чтобы не рисовать
 * разделы, которые всё равно ответят 403. Поэтому «роль неизвестна» приравнено
 * к «прав нет» — ошибка в эту сторону лишь прячет пункт меню, а в обратную
 * показала бы админскую форму тому, кто ею не пользуется.
 *
 * Подсказка живёт не дольше access-токена. Если она протухла, роль
 * запрашивается у сервера обменом токенов; на это время isResolving = true,
 * чтобы заглушка «нет доступа» не мигала перед законным админом.
 */
/**
 * Роли, которым положено управлять содержимым: заказ презентаций, а в
 * дальнейшем и всё остальное, что делает контент-менеджер.
 *
 * Админ входит сюда не «заодно»: на сервере он старше контент-менеджера и
 * проходит те же проверки (backend/app/api/deps.py), поэтому интерфейс, не
 * пускающий админа в раздел контент-менеджера, врал бы о правах.
 */
const CONTENT_MANAGEMENT_ROLES = ['admin', 'content_manager'];

export const useSessionRole = () => {
    const [role, setRole] = useState(() => getSessionRole());
    const [isResolving, setIsResolving] = useState(() => !getSessionRole() && hasActiveSession());

    useEffect(() => {
        let cancelled = false;

        const syncRole = () => setRole(getSessionRole());
        const unsubscribe = subscribeSessionRole(syncRole);

        // Начальные значения уже посчитаны инициализаторами состояния, здесь
        // остаётся только добрать роль у сервера, если подсказки нет или она
        // истекла вместе с access-токеном.
        if (!getSessionRole() && hasActiveSession()) {
            ensureSessionRole()
                .catch(() => '')
                .then((resolved) => {
                    if (cancelled) return;
                    setRole(resolved || '');
                    setIsResolving(false);
                });
        }

        return () => {
            cancelled = true;
            unsubscribe();
        };
    }, []);

    return {
        role,
        isResolving,
        isAdmin: role === 'admin',
        // Неизвестная роль (пустая строка) сюда не попадает — см. пояснение
        // выше: «роль не подтверждена» трактуется как «прав нет».
        canManageContent: CONTENT_MANAGEMENT_ROLES.includes(role),
    };
};

export default useSessionRole;
