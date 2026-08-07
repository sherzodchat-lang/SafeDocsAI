import { useCallback, useMemo, useState } from 'react';

const EMPTY_STATE = Object.freeze({ scope: null, opened: Object.freeze({}) });

/**
 * Какие папки раскрыты.
 *
 * Хранятся не сами папки, а только ЯВНЫЕ нажатия пользователя: у закрытого и
 * открытого по умолчанию состояния разные причины (мало документов — открыты,
 * идёт поиск — открыты те, где нашлось), и записать умолчание в состояние
 * значит потерять момент, когда причина изменилась.
 *
 * scope — то, при чём эти нажатия сделаны (обычно строка поиска). Со сменой
 * scope нажатия перестают действовать: папку закрыли под один запрос, а под
 * следующим она обязана открыться снова, иначе поиск молча прячет найденное.
 * Сравнение вместо очистки по useEffect — чтобы не было кадра, в котором
 * состояние уже от нового запроса, а разметка ещё от старого.
 *
 * Хук общий для раздела «Темы» и выбора источников: папки в двух местах одни и
 * те же, и раскрываться по-разному им нельзя.
 */
export const useFolderDisclosure = (scope) => {
    const [state, setState] = useState(EMPTY_STATE);

    const opened = state.scope === scope ? state.opened : EMPTY_STATE.opened;

    const isOpen = useCallback(
        (key, fallback) => opened[key] ?? fallback,
        [opened],
    );

    // Новое значение приходит снаружи, а не считается как «наоборот»: умолчание
    // известно только на месте отрисовки, и вычислять его второй раз внутри
    // обновления состояния — верный способ разойтись с тем, что видно на экране.
    const setOpen = useCallback((key, open) => {
        setState((previous) => ({
            scope,
            opened: { ...(previous.scope === scope ? previous.opened : null), [key]: open },
        }));
    }, [scope]);

    return useMemo(() => ({ isOpen, setOpen }), [isOpen, setOpen]);
};

export default useFolderDisclosure;
