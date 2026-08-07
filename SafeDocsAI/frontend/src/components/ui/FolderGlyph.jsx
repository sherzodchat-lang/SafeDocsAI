import React from 'react';
import { Folder, FolderClock, FolderOpen, FolderSearch } from 'lucide-react';

import { cn } from '../../lib/utils';
import { TOPIC_FOLDER_TOPIC, TOPIC_FOLDER_UNCLEAR } from '../../lib/topics';

/**
 * Значок папки темы — один на раздел «Темы» и на выбор источников для блокнота.
 *
 * Три вида различаются нарочно: с одинаковыми значками особую папку от обычной
 * темы отличала бы только подпись, а её ещё надо прочитать. Лупа — модель искала
 * тему и не нашла, часы — очередь на разметку, открытая папка — раскрытая тема.
 *
 * Отдельный компонент, а не функция «kind → иконка» в каждом экране: одинаковые
 * папки в двух местах обязаны выглядеть одинаково, а разъезжаются такие таблицы
 * первыми.
 */
const FolderGlyph = ({ folder, isOpen = false, className }) => {
    const isTopic = folder?.kind === TOPIC_FOLDER_TOPIC;
    const Icon = folder?.kind === TOPIC_FOLDER_UNCLEAR
        ? FolderSearch
        : !isTopic
            ? FolderClock
            : isOpen ? FolderOpen : Folder;

    return (
        <Icon
            className={cn(
                'h-4 w-4 shrink-0',
                isTopic ? 'text-[#1f3a60]' : 'text-slate-400',
                className,
            )}
            aria-hidden="true"
        />
    );
};

export default FolderGlyph;
