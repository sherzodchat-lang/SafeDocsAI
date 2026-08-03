import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Pencil, X } from 'lucide-react';

import { Button } from '../ui/Button';
import Input from '../ui/Input';
import { useModalDialog } from '../../hooks/useModalDialog';
import { useLocale } from '../../i18n';
import { resolveApiErrorMessage } from '../../lib/apiError';
import { notebooksService } from '../../services/notebooksService';

// Те же три профиля, что предлагает форма создания блокнота: чужое значение
// бэкенд отклонит кодом source.unsupported_domain_profile.
const DOMAIN_PROFILES = ['general', 'tax', 'legal'];

/**
 * Правка блокнота: имя, описание и профиль предметной области.
 *
 * Один диалог на список блокнотов и на шапку внутри блокнота: PATCH частичный,
 * и собирать тело по двум разным правилам в двух местах значило бы дважды ловить
 * source.nothing_to_update и дважды объяснять, что описание чистится через null.
 * Обновлённый блокнот уходит вызывающему через onSaved — состояние живёт там же,
 * где список или шапка, поэтому имя меняется без перезагрузки страницы.
 */
const NotebookEditDialog = ({ notebook, isOpen, onClose, onSaved }) => {
  const { t } = useLocale();
  const dialogRef = useRef(null);
  const nameInputRef = useRef(null);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [domainProfile, setDomainProfile] = useState('general');
  const [isSaving, setIsSaving] = useState(false);
  const [error, setError] = useState('');

  // Поля заполняет открытие диалога: пока он закрыт, набранное для прошлого
  // блокнота не должно переезжать на следующий.
  useEffect(() => {
    if (!isOpen) return;

    setName(notebook?.name || '');
    setDescription(notebook?.description || '');
    setDomainProfile(notebook?.domain_profile || 'general');
    setError('');
  }, [isOpen, notebook]);

  // Закрытие по Escape и клику по подложке отключаем на время запроса: ответ
  // всё равно придёт, а пользователь остался бы без сообщения об ошибке.
  const handleClose = useCallback(() => {
    if (isSaving) return;
    onClose();
  }, [isSaving, onClose]);

  useModalDialog(isOpen, handleClose, dialogRef, nameInputRef);

  // Отправляем только изменённое: PATCH со всеми полями подряд без нужды
  // переписывал бы описание и профиль, а пустое тело бэкенд отвергает кодом
  // source.nothing_to_update.
  const changes = useMemo(() => {
    if (!notebook) return {};

    const payload = {};
    const nextName = name.trim();
    const nextDescription = description.trim();

    if (nextName && nextName !== (notebook.name || '')) payload.name = nextName;
    // Пустое поле — это очистка описания, а её бэкенд принимает только как null.
    if (nextDescription !== (notebook.description || '')) payload.description = nextDescription || null;
    if (domainProfile !== (notebook.domain_profile || 'general')) payload.domain_profile = domainProfile;

    return payload;
  }, [notebook, name, description, domainProfile]);

  const hasChanges = Object.keys(changes).length > 0;

  const getDomainProfileLabel = (value) => {
    const key = `notebooksPage.profiles.${value}`;
    const translated = t(key);
    return translated === key ? value : translated;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    // Второй клик до ответа сервера отправлял бы тот же PATCH ещё раз.
    if (isSaving || !notebook) return;

    if (!name.trim()) {
      setError(t('notebookEdit.nameRequired'));
      nameInputRef.current?.focus();
      return;
    }

    if (!hasChanges) {
      onClose();
      return;
    }

    try {
      setIsSaving(true);
      setError('');
      const response = await notebooksService.update(notebook.id, changes);
      onSaved?.(response.data);
      onClose();
    } catch (submitError) {
      console.error('Failed to update notebook', submitError);
      setError(resolveApiErrorMessage(submitError, t, 'notebookEdit.saveFailed'));
    } finally {
      setIsSaving(false);
    }
  };

  if (!isOpen || !notebook) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div className="absolute inset-0 bg-black/40 backdrop-blur-[2px]" onClick={handleClose} aria-hidden="true" />

      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        aria-label={t('notebookEdit.title')}
        className="relative flex w-full max-w-lg flex-col overflow-hidden rounded-2xl bg-white shadow-2xl"
      >
        <div className="flex items-start justify-between gap-4 px-6 py-5">
          <div>
            <div className="flex items-center gap-2">
              <Pencil className="h-4 w-4 text-[#1f3a60]" />
              <p className="text-[15px] font-semibold text-slate-900">{t('notebookEdit.title')}</p>
            </div>
            <p className="mt-1 text-sm text-slate-500">{t('notebookEdit.description')}</p>
          </div>
          <button
            type="button"
            onClick={handleClose}
            disabled={isSaving}
            className="inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-slate-400 transition hover:bg-slate-100 hover:text-slate-700 disabled:pointer-events-none disabled:opacity-60"
            aria-label={t('notebookEdit.close')}
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <form onSubmit={handleSubmit} className="flex flex-col gap-4 px-6 pb-6">
          <label className="flex flex-col gap-1.5 text-sm font-semibold text-slate-700">
            {t('notebookEdit.nameLabel')}
            <Input
              ref={nameInputRef}
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder={t('notebookEdit.namePlaceholder')}
              disabled={isSaving}
            />
          </label>

          <label className="flex flex-col gap-1.5 text-sm font-semibold text-slate-700">
            {t('notebookEdit.descriptionLabel')}
            <textarea
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              placeholder={t('notebookEdit.descriptionPlaceholder')}
              rows={4}
              disabled={isSaving}
              className="w-full resize-none rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-normal text-slate-800 placeholder-slate-400 transition focus:border-[#1f3a60] focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/20 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </label>

          <label className="flex flex-col gap-1.5 text-sm font-semibold text-slate-700">
            {t('notebookEdit.profileLabel')}
            <select
              value={domainProfile}
              onChange={(event) => setDomainProfile(event.target.value)}
              disabled={isSaving}
              className="h-10 rounded-lg border border-slate-300 bg-white px-3 text-sm font-normal text-slate-700 transition focus:border-[#1f3a60] focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/20 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {DOMAIN_PROFILES.map((profile) => (
                <option key={profile} value={profile}>{getDomainProfileLabel(profile)}</option>
              ))}
            </select>
          </label>

          {error ? (
            <div role="alert" className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600">
              {error}
            </div>
          ) : null}

          <div className="flex items-center justify-end gap-3 border-t border-slate-200 pt-4">
            <Button type="button" variant="ghost" onClick={handleClose} disabled={isSaving}>
              {t('notebookEdit.cancel')}
            </Button>
            <Button type="submit" isLoading={isSaving} disabled={!name.trim() || !hasChanges}>
              {isSaving ? t('notebookEdit.saving') : t('notebookEdit.save')}
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default NotebookEditDialog;
