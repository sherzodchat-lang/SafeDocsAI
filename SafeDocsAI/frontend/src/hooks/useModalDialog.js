import { useEffect } from 'react';

const FOCUSABLE_SELECTOR = 'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])';

/**
 * Поведение модального диалога: фокус переносится внутрь при открытии и
 * возвращается на элемент-инициатор при закрытии, Escape закрывает, Tab не уходит за пределы.
 */
export const useModalDialog = (isOpen, onClose, containerRef, initialFocusRef) => {
    useEffect(() => {
        if (!isOpen) return undefined;

        const previouslyFocused = document.activeElement;
        initialFocusRef?.current?.focus();

        const handleKeyDown = (event) => {
            if (event.key === 'Escape') {
                event.preventDefault();
                onClose();
                return;
            }

            if (event.key !== 'Tab') return;

            const container = containerRef.current;
            if (!container) return;

            const focusable = Array.from(container.querySelectorAll(FOCUSABLE_SELECTOR))
                .filter((element) => element.offsetParent !== null);
            if (focusable.length === 0) return;

            const first = focusable[0];
            const last = focusable[focusable.length - 1];

            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        };

        document.addEventListener('keydown', handleKeyDown);

        return () => {
            document.removeEventListener('keydown', handleKeyDown);
            // Инициатор мог исчезнуть вместе со строкой (например, источник удалён).
            if (previouslyFocused instanceof HTMLElement && document.contains(previouslyFocused)) {
                previouslyFocused.focus();
            }
        };
    }, [isOpen, onClose, containerRef, initialFocusRef]);
};

export default useModalDialog;
