import React, { useEffect, useMemo, useState } from 'react';
import { Link, Outlet, useLocation, useNavigate, useSearchParams } from 'react-router-dom';
import {
    ChevronLeft,
    ChevronRight,
    Bookmark,
    CalendarDays,
    ChartNoAxesCombined,
    Clock3,
    FileText,
    LogOut,
    MessageSquare,
    Pencil,
    Settings,
    Shapes,
    Trash2,
    UserCircle2,
} from 'lucide-react';
import { Button } from '../ui/Button';
import { getSessionUsername, logout } from '../../services/api';
import { cn } from '../../lib/utils';
import { NotebookHeaderContext } from './NotebookHeaderContext';
import LocaleSwitcher from '../i18n/LocaleSwitcher';
import { useLocale } from '../../i18n';
import { useSessionRole } from '../../hooks/useSessionRole';

const resolvePageMeta = (pathname, t) => {
    if (pathname === '/' || pathname.startsWith('/chat')) {
        return { title: t('layout.pageMeta.homeTitle'), badge: t('layout.pageMeta.homeBadge'), searchPlaceholder: t('layout.pageMeta.homeSearch') };
    }
    if (pathname.startsWith('/ask')) {
        return { title: t('layout.pageMeta.askTitle'), badge: t('layout.pageMeta.askBadge'), searchPlaceholder: t('layout.pageMeta.askSearch') };
    }
    if (pathname.startsWith('/notes')) {
        return { title: t('layout.pageMeta.notesTitle'), badge: t('layout.pageMeta.notesBadge'), searchPlaceholder: t('layout.pageMeta.notesSearch') };
    }
    if (pathname.startsWith('/topics')) {
        return { title: t('layout.pageMeta.topicsTitle'), badge: t('layout.pageMeta.topicsBadge'), searchPlaceholder: t('layout.pageMeta.topicsSearch') };
    }
    if (pathname.startsWith('/insights')) {
        return { title: t('layout.pageMeta.insightsTitle'), badge: t('layout.pageMeta.insightsBadge'), searchPlaceholder: t('layout.pageMeta.insightsSearch') };
    }
    if (/^\/notebooks\/[^/]+(\/|$)/.test(pathname)) {
        return { title: t('layout.pageMeta.notebookTitle'), badge: t('layout.pageMeta.notebookBadge'), searchPlaceholder: t('layout.pageMeta.notebookSearch') };
    }
    if (pathname.startsWith('/notebooks')) {
        return { title: t('layout.pageMeta.notebooksTitle'), badge: t('layout.pageMeta.notebooksBadge'), searchPlaceholder: t('layout.pageMeta.notebooksSearch') };
    }
    if (pathname.startsWith('/sources') || pathname.startsWith('/admin/sources') || pathname.startsWith('/admin/documents')) {
        return { title: t('layout.pageMeta.sourcesTitle'), badge: t('layout.pageMeta.sourcesBadge'), searchPlaceholder: t('layout.pageMeta.sourcesSearch') };
    }
    if (pathname.startsWith('/admin/logs')) {
        return { title: t('layout.pageMeta.logsTitle'), badge: t('layout.pageMeta.logsBadge'), searchPlaceholder: t('layout.pageMeta.logsSearch') };
    }
    if (pathname.startsWith('/settings')) {
        return { title: t('layout.pageMeta.settingsTitle'), badge: t('layout.pageMeta.settingsBadge'), searchPlaceholder: t('layout.pageMeta.settingsSearch') };
    }

    return { title: t('layout.pageMeta.homeTitle'), badge: t('layout.pageMeta.fallbackBadge'), searchPlaceholder: t('layout.pageMeta.fallbackSearch') };
};

const isActiveLink = (pathname, href) => pathname === href || (href !== '/' && pathname.startsWith(`${href}/`));

const isNotebookDetailRoute = (pathname) => /^\/notebooks\/[^/]+(?:\/.*)?$/.test(pathname);

const Layout = () => {
    const location = useLocation();
    const navigate = useNavigate();
    const [searchParams, setSearchParams] = useSearchParams();
    const { t } = useLocale();
    const { isAdmin } = useSessionRole();
    // Журнал и настройки бэкенд отдаёт только админу, поэтому остальным они и
    // не показываются: пункт меню, ведущий на заглушку «нет доступа», — не
    // защита, а лишний тупик. Сама защита остаётся на сервере.
    const navItems = useMemo(() => ([
        { name: t('layout.nav.sources'), href: '/sources', icon: FileText },
        // Темы стоят рядом с источниками: это второй способ смотреть на тот же
        // список, и из него же в него и ведёт переход по теме.
        { name: t('layout.nav.topics'), href: '/topics', icon: Shapes },
        { name: t('layout.nav.notebooks'), href: '/notebooks', icon: Bookmark },
        { name: t('layout.nav.chat'), href: '/chat', icon: MessageSquare },
        { name: t('layout.nav.logs'), href: '/admin/logs', icon: ChartNoAxesCombined, adminOnly: true },
        { name: t('layout.nav.settings'), href: '/settings', icon: Settings, adminOnly: true },
    ].filter((item) => !item.adminOnly || isAdmin)), [t, isAdmin]);

    // Имя больше не достать из JWT: access-токен лежит в httpOnly-куке. Берём
    // несекретное имя, запомненное при входе.
    const userInfo = useMemo(() => {
        const username = getSessionUsername();
        if (!username) return { username: t('layout.user.defaultName'), email: '' };
        return {
            username,
            email: `${username}@knowledge.local`,
        };
    }, [t]);

    const [notebookHeader, setNotebookHeader] = useState(null);
    const [notebookActions, setNotebookActions] = useState(null);
    const [notebookTabs, setNotebookTabs] = useState(null);
    const [sidebarCollapsed, setSidebarCollapsed] = useState(() => localStorage.getItem('knowledgeai.sidebarCollapsed') === 'true');
    const [isLoggingOut, setIsLoggingOut] = useState(false);

    const searchValue = searchParams.get('q') || '';

    // Правим только свой параметр: адрес страницы может нести и её собственное
    // состояние (фильтр по теме), а набор текста в поиске не должен его сбивать.
    // replace — чтобы каждая буква не оставляла шаг в истории браузера.
    const handleSearchChange = (e) => {
        const val = e.target.value;
        const params = new URLSearchParams(searchParams);

        if (val) {
            params.set('q', val);
        } else {
            params.delete('q');
        }

        setSearchParams(params, { replace: true });
    };

    // Поле поиска принадлежит странице, поэтому на новой странице оно пустое.
    // Гасим ровно свой параметр, а не адрес целиком: остальные параметры —
    // состояние самой страницы (например, фильтр по теме, с которым на список
    // источников приходят с экрана «Темы»), и чистка «всего» молча снимала бы
    // фильтр в тот же миг, когда пользователь его поставил.
    useEffect(() => {
        const params = new URLSearchParams(location.search);
        if (!params.has('q')) return;

        params.delete('q');
        // replace: иначе в истории остаётся лишний шаг и «Назад» возвращает на ту же страницу.
        setSearchParams(params, { replace: true });
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [location.pathname]);

    const pageMeta = resolvePageMeta(location.pathname, t);
    const notebookDetailRoute = isNotebookDetailRoute(location.pathname);

    // Гасим refresh на бэкенде: без этого выданный токен остаётся рабочим ещё
    // неделю, сколько бы раз пользователь ни нажал «Выход».
    const handleLogout = async () => {
        if (isLoggingOut) return;

        setIsLoggingOut(true);
        await logout();
        navigate('/login', { replace: true });
    };

    const handleToggleSidebar = () => {
        setSidebarCollapsed((prev) => {
            const next = !prev;
            localStorage.setItem('knowledgeai.sidebarCollapsed', String(next));
            return next;
        });
    };

    return (
        <NotebookHeaderContext.Provider value={{ notebookHeader, setNotebookHeader, notebookActions, setNotebookActions, notebookTabs, setNotebookTabs }}>
        <div className="min-h-screen bg-[#f3f5f8] lg:flex lg:h-screen lg:overflow-hidden">
            <aside className={cn(
                'hidden flex-col bg-[#1f3a60] text-white transition-[width] duration-300 lg:flex lg:h-screen',
                sidebarCollapsed ? 'w-[84px]' : 'w-64'
            )}>
                <div className={cn('flex h-16 items-center border-b border-white/10', sidebarCollapsed ? 'justify-center px-3' : 'gap-3 px-5')}>

                    {!sidebarCollapsed ? (
                    <div>
                        <p className="text-2xl font-bold leading-none">SafeDocsAI</p>
                    </div>
                    ) : null}
                </div>

                <div className={cn('border-b border-white/10 py-3', sidebarCollapsed ? 'px-3' : 'px-4')}>
                    <button
                        type="button"
                        onClick={handleToggleSidebar}
                        className={cn(
                            'flex w-full items-center rounded-lg border border-white/15 text-sm font-semibold text-slate-200 transition hover:bg-white/10 hover:text-white',
                            sidebarCollapsed ? 'justify-center px-0 py-2.5' : 'justify-between px-3 py-2.5'
                        )}
                        aria-label={sidebarCollapsed ? t('layout.actions.expandSidebar') : t('layout.actions.collapseSidebar')}
                    >
                        {!sidebarCollapsed ? <span>{t('layout.actions.hideMenu')}</span> : null}
                        {sidebarCollapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
                    </button>
                </div>

                <nav className={cn('flex-1 space-y-1 py-5', sidebarCollapsed ? 'px-2' : 'px-3')}>
                    {navItems.map((item) => {
                        const Icon = item.icon;
                        const active = isActiveLink(location.pathname, item.href);

                        return (
                            <Link
                                key={item.name}
                                to={item.href}
                                className={cn(
                                    'group flex items-center rounded-lg py-2.5 text-sm font-semibold transition-all',
                                    sidebarCollapsed ? 'justify-center px-2' : 'gap-3 px-3',
                                    active
                                        ? 'bg-white/12 text-white'
                                        : 'text-slate-200/85 hover:bg-white/8 hover:text-white',
                                )}
                                title={sidebarCollapsed ? item.name : undefined}
                            >
                                <Icon className={cn('h-[18px] w-[18px]', active ? 'text-white' : 'text-slate-300 group-hover:text-white')} />
                                {!sidebarCollapsed ? item.name : null}
                            </Link>
                        );
                    })}
                </nav>

                <div className={cn('border-t border-white/10', sidebarCollapsed ? 'p-2' : 'p-4')}>
                    <div className={cn('mb-3 rounded-xl bg-white/5', sidebarCollapsed ? 'flex justify-center p-2.5' : 'flex items-center gap-3 p-2.5')}>
                        <UserCircle2 className="h-9 w-9 text-slate-300" />
                        {!sidebarCollapsed ? (
                        <div className="min-w-0">
                            <p className="truncate text-sm font-semibold text-white">{userInfo.username}</p>
                            <p className="truncate text-xs text-slate-300">{userInfo.email}</p>
                        </div>
                        ) : null}
                    </div>
                    <button
                        type="button"
                        onClick={handleLogout}
                        disabled={isLoggingOut}
                        className={cn(
                            'flex w-full items-center justify-center rounded-lg border border-white/20 text-sm font-semibold text-white transition hover:bg-white/10 disabled:opacity-60',
                            sidebarCollapsed ? 'px-0 py-2.5' : 'gap-2 px-3 py-2'
                        )}
                        title={sidebarCollapsed ? t('layout.actions.logout') : undefined}
                    >
                        <LogOut className="h-4 w-4" />
                        {!sidebarCollapsed ? (isLoggingOut ? t('layout.actions.loggingOut') : t('layout.actions.logout')) : null}
                    </button>
                </div>
            </aside>

            <div className="flex min-h-screen flex-1 flex-col lg:h-screen">
                {/* Вкладки блокнота стоят в самой шапке, а не блоком под ней:
                    нижняя граница шапки служит им направляющей, и экран не
                    начинается с двух полос навигации подряд. */}
                <header className={cn(
                    'border-b border-slate-200 bg-white px-4 sm:px-6 lg:px-8',
                    notebookTabs && notebookTabs.items.length > 0 ? 'pt-3' : 'py-3',
                )}>
                    <div className="flex flex-wrap items-center justify-between gap-3">
                        {/* min-w-0 обязателен: без него flex-элемент не сжимается ниже длины
                            содержимого, и truncate у имени блокнота (до 255 символов) не работает. */}
                        <div className="flex min-w-0 items-center gap-3">
                            <div className="lg:hidden flex h-8 w-8 items-center justify-center rounded-lg bg-[#1f3a60] text-sm font-extrabold text-[#c5a059]">
                                S
                            </div>
                            {!notebookDetailRoute ? (
                                <div className="flex items-center gap-3">
                                    <h1 className="text-xl lg:text-2xl font-extrabold text-[#1f3a60]">{pageMeta.title}</h1>
                                    <span className="rounded-full bg-[#1f3a60]/10 px-2.5 py-0.5 text-[10px] lg:text-xs font-bold text-[#1f3a60]">
                                        {pageMeta.badge}
                                    </span>
                                </div>
                            ) : notebookHeader ? (
                                /* Описание, даты и профиль стоят одной строкой под именем:
                                   тремя отдельными строками шапка занимала треть высоты
                                   экрана, а данные в ней справочные — их читают редко, но
                                   убрать нельзя. */
                                <div className="flex min-w-0 flex-col gap-0.5">
                                    <div className="flex min-w-0 items-center gap-2">
                                        <h1 className="truncate text-lg lg:text-2xl font-extrabold text-[#1f3a60]" title={notebookHeader.name}>
                                            {notebookHeader.name}
                                        </h1>
                                        {/* Слово «Профиль» ушло в подсказку: в строке с именем
                                            блокнота оно занимало место, а значение говорит само
                                            за себя. */}
                                        <span
                                            title={`${t('layout.actions.profile')}: ${notebookHeader.domainProfile}`}
                                            className="inline-flex shrink-0 items-center gap-1 rounded-full bg-[#1f3a60]/10 px-2.5 py-0.5 text-[11px] font-bold text-[#1f3a60]"
                                        >
                                            <Bookmark className="h-3 w-3" />
                                            {notebookHeader.domainProfile}
                                        </span>
                                    </div>
                                    <div className="flex min-w-0 flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] font-medium text-slate-400">
                                        {notebookHeader.description ? (
                                            <span className="max-w-[22rem] truncate text-slate-500" title={notebookHeader.description}>
                                                {notebookHeader.description}
                                            </span>
                                        ) : null}
                                        <span className="inline-flex items-center gap-1.5">
                                            <CalendarDays className="h-3.5 w-3.5" />
                                            {t('layout.actions.created')}: {notebookHeader.createdAtText}
                                        </span>
                                        <span className="inline-flex items-center gap-1.5">
                                            <Clock3 className="h-3.5 w-3.5" />
                                            {t('layout.actions.updated')}: {notebookHeader.updatedAtText}
                                        </span>
                                    </div>
                                </div>
                            ) : null}
                        </div>

                        <div className="flex w-full items-center gap-3 sm:w-auto">
                            {!notebookDetailRoute ? (
                                <>
                                    <LocaleSwitcher className="mr-2" buttonClassName="px-2 py-0.5 text-[10px] font-bold" />

                                    <div className="relative flex-1 sm:w-64 sm:flex-none">
                                        <input
                                            type="text"
                                            value={searchValue}
                                            onChange={handleSearchChange}
                                            placeholder={pageMeta.searchPlaceholder}
                                            className="h-9 w-full rounded-lg border border-slate-200 bg-slate-50 px-3 text-sm text-slate-700 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-[#1f3a60]/30"
                                        />
                                    </div>
                                </>
                            ) : (
                                <>
                                    {/* Внутри блокнота переключатель тоже нужен: продукт двуязычный,
                                        иначе язык меняется только уходом в другой раздел и обратно. */}
                                    <LocaleSwitcher className="mr-2" buttonClassName="px-2 py-0.5 text-[10px] font-bold" />

                                    {notebookActions ? (
                                        <div className="flex flex-wrap items-center justify-end gap-2">
                                            {/* Область глобального чата задаётся только этой кнопкой.
                                                Заливка и подпись показывают текущее состояние: у уже
                                                активного блокнота повторное назначение бессмысленно,
                                                поэтому клик по нему возвращает чат ко всем источникам. */}
                                            {notebookActions.onToggleActiveForChat ? (
                                                <Button
                                                    type="button"
                                                    variant={notebookActions.isActiveForChat ? 'primary' : 'outline'}
                                                    className="justify-center"
                                                    disabled={notebookActions.toggleActiveForChatDisabled}
                                                    title={notebookActions.toggleActiveForChatTitle}
                                                    aria-pressed={notebookActions.isActiveForChat}
                                                    onClick={notebookActions.onToggleActiveForChat}
                                                >
                                                    <MessageSquare className="h-4 w-4" />
                                                    {notebookActions.isActiveForChat
                                                        ? t('layout.actions.activeForChat')
                                                        : t('layout.actions.makeActiveForChat')}
                                                </Button>
                                            ) : null}
                                            {/* Правка и удаление — обслуживание блокнота, а не работа
                                                в нём: они остались рядом, но значками. Тремя полными
                                                кнопками шапка спорила с единственным действием,
                                                которое здесь и правда выбирают, — областью чата.
                                                Названия действий целиком остались в подсказке и в
                                                метке для скринридера. */}
                                            {notebookActions.onEdit ? (
                                                <Button
                                                    type="button"
                                                    variant="outline"
                                                    size="icon"
                                                    disabled={notebookActions.editDisabled}
                                                    title={notebookActions.editTitle}
                                                    aria-label={notebookActions.editTitle}
                                                    onClick={notebookActions.onEdit}
                                                >
                                                    <Pencil className="h-4 w-4" />
                                                </Button>
                                            ) : null}
                                            <Button
                                                type="button"
                                                variant="outline"
                                                size="icon"
                                                className="text-red-600 hover:border-red-300 hover:bg-red-50 hover:text-red-700"
                                                disabled={notebookActions.deleteDisabled}
                                                title={notebookActions.deleteTitle}
                                                aria-label={notebookActions.deleteTitle}
                                                onClick={notebookActions.onDelete}
                                            >
                                                <Trash2 className="h-4 w-4" />
                                            </Button>
                                        </div>
                                    ) : null}
                                </>
                            )}

                            <button
                                type="button"
                                onClick={handleLogout}
                                disabled={isLoggingOut}
                                className="inline-flex h-9 w-9 items-center justify-center rounded-lg border border-slate-200 bg-white text-slate-500 transition hover:bg-slate-50 hover:text-red-600 disabled:opacity-60 lg:hidden"
                                aria-label={t('layout.actions.logoutMobile')}
                            >
                                <LogOut className="h-4 w-4" />
                            </button>
                        </div>
                    </div>

                    <div className="mt-3 flex flex-wrap gap-2 lg:hidden">
                        {navItems.map((item) => (
                            <Link
                                key={item.name}
                                to={item.href}
                                className={cn(
                                    'rounded-lg border px-3 py-1 text-[11px] font-semibold',
                                    isActiveLink(location.pathname, item.href)
                                        ? 'border-[#1f3a60] bg-[#1f3a60] text-white'
                                        : 'border-slate-300 bg-white text-slate-600'
                                )}
                            >
                                {item.name}
                            </Link>
                        ))}
                    </div>

                    {notebookTabs && notebookTabs.items.length > 0 ? (
                        <nav aria-label={notebookTabs.label} className="-mb-px mt-3 flex flex-wrap">
                            {notebookTabs.items.map((tab) => (
                                <Link
                                    key={tab.key}
                                    to={tab.href}
                                    aria-current={tab.isActive ? 'page' : undefined}
                                    className={cn(
                                        'border-b-2 px-3 py-2 text-sm font-semibold transition focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-[#1f3a60]/40',
                                        tab.isActive
                                            ? 'border-[#1f3a60] text-[#1f3a60]'
                                            : 'border-transparent text-slate-500 hover:text-[#1f3a60]',
                                    )}
                                >
                                    {tab.label}
                                </Link>
                            ))}
                        </nav>
                    ) : null}
                </header>

                <main className="soft-grid flex-1 overflow-auto p-6">
                    <Outlet />
                </main>
            </div>
        </div>
        </NotebookHeaderContext.Provider>
    );
};

export default Layout;
